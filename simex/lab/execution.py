"""Serialize physics and make execution/result commits independently recoverable."""
from __future__ import annotations

import contextlib
import fcntl
import importlib.metadata
import json
from pathlib import Path

from .records import Journal, immutable, read, save, sha
from .. import research as r
from ..errors import ResearchError
from ..source_model import model_hash
from ..runner import Runner
from ..stage2 import check_rollout, seal_rollout


class TrackedRunner:
    def __init__(self, delegate, run, repetition, config, *, refresh=None, fault=None):
        self.delegate = delegate
        self.run_dir, self.repetition = Path(run), Path(repetition)
        self.config, self.agent = config, delegate.agent
        self.journal = Journal(run)
        self.source_model = self.repetition / 'source_models/initial'
        self.refresh, self.fault = refresh, fault

    def _inject(self, event):
        if self.fault:
            self.fault(event)

    def prepare_policy(self, *args, **kwargs):
        return self.delegate.prepare_policy(*args, **kwargs)

    def rollout(self, engine, task, toolbox, out, script=None, source_config=None, source_model=None):
        out, toolbox = Path(out).resolve(), Path(toolbox).resolve()
        if not out.is_relative_to(self.run_dir.resolve()):
            raise ResearchError('Execution artifacts must remain inside the campaign')
        if (self.run_dir / 'STOP').exists():
            raise ResearchError('Campaign stop requested')
        state = read(self.run_dir / 'state.json', {})
        save(self.run_dir / 'state.json', {**state, 'active': str(out.relative_to(self.run_dir.resolve())),
                                         'active_task': task['name']})
        if engine == 'isaac' and self.config['offline_acceptance'] and not getattr(self.delegate, 'simulated', False):
            raise ResearchError('Offline acceptance forbids Isaac target execution')
        source_model = (source_model or self.source_model) if engine == 'mujoco' else None
        from ..source_model import check_model
        if source_model is not None:
            check_model(source_model)
        r._validate_toolbox(r._contents(toolbox))
        # Policies are prepared before any physical operation is allocated. A
        # generation disconnect therefore leaves a resumable, unstarted slot.
        self._inject('before_policy')
        if script is None:
            prepared = self.delegate.prepare_policy(task, toolbox, out)
            if prepared and prepared.get('status') == 'generation_error':
                raise ResearchError('Policy generation failed before simulator startup: ' +
                                    prepared.get('reason', 'see policy_generation.json'))
        self._inject('before_execution')
        ident = 'physics/' + str(out.relative_to(self.run_dir.resolve()))
        request = {'engine': engine, 'task_sha256': sha(task), 'library_sha256': r.toolbox_hash(toolbox),
                   'model_sha256': model_hash(source_model) if source_model else None,
                   'script_sha256': sha(script if script is not None else read(out / 'policy_generation.json')),
                   'score_contract': self.config['placement_contract'],
                   'simulated': bool(getattr(self.delegate, 'simulated', False))}
        context_model = getattr(self, 'target_source_context', None) if engine == 'isaac' else None
        request['source_context_sha256'] = model_hash(context_model) if context_model else None
        # There is one advisory GPU lock shared by every campaign and source/target
        # worker. Only inference preparation can overlap, never physical tests.
        lock_path = Path('/tmp/simex-physics.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            old, fresh = self.journal.begin(ident, engine, request)
            previous = read(out / 'result.json')
            if not fresh:
                if old['state'] == 'complete':
                    check_rollout(out)
                    result = json.loads(old['result'])
                    if previous != result:
                        raise ResearchError('Durable result does not match operation ledger')
                    return result
                # Commit interruption after saving the result can be reconciled;
                # an execution without a durable result is never invoked twice.
                if previous is not None and (out / 'integrity.json').exists():
                    check_rollout(out)
                    self.journal.finish(ident, previous)
                    return previous
                raise ResearchError('Interrupted physical operation; refusing automatic rerun: ' + ident)
            immutable(out / 'operation_input.json', request)
            try:
                self._inject('execution_allocated')
                result = self.delegate.rollout(engine, task, toolbox, out, script=script,
                                               source_config=source_config, source_model=source_model)
                self._inject('after_execution')
                if result.get('actual_started') and not request['simulated']:
                    evidence = result.get('public_evidence', {})
                    required = [out / 'policy.py', out / 'public/overhead.mp4',
                                out / 'public/robot_trace.jsonl', out / 'public/actions.jsonl']
                    missing = [str(path.relative_to(out)) for path in required if not path.is_file()]
                    if missing or evidence.get('recording_errors'):
                        save(out / 'unverified_result.json', result)
                        result = {**result, 'success': False, 'status': 'evidence_error',
                                  'failure_kind': 'infrastructure', 'reason': 'Required public evidence missing',
                                  'missing_evidence': missing, 'evidence_rejected': True}
                result['simulated'] = request['simulated']
                result['engine'] = 'mock_' + engine if result['simulated'] else engine
                if engine == 'mujoco':
                    version = importlib.metadata.version('mujoco')
                else:
                    version = result.get('engine_version', 'unreported')
                result.update(engine_version=version, score_version=self.config['placement_contract'],
                    library_sha256=request['library_sha256'], source_model_sha256=request['model_sha256'],
                    source_context_sha256=request['source_context_sha256'])
                r._freeze(toolbox, out / 'executed_library')
                save(out / 'result.json', result)
                # _run creates research_request before reaching this layer.
                if not (out / 'research_request.json').exists():
                    save(out / 'research_request.json', request)
                seal_rollout(out)
                self._inject('after_result_save')
                self.journal.finish(ident, result)
                self._inject('after_commit')
                return result
            except BaseException as exc:
                save(out / 'interruption.json', {'type': type(exc).__name__, 'message': str(exc),
                     'operation': ident, 'physical_rerun_permitted': False})
                raise
            finally:
                if self.refresh:
                    self.refresh()

    def close(self):
        self.delegate.close()
