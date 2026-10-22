"""Campaign lifecycle. Formal starts always require an explicit fresh request cap."""
from __future__ import annotations

from pathlib import Path
import signal

from . import PROTOCOL
from .protocol import initialize, check_resume, ROOT
from .records import immutable, read, save
from .gateway import Gateway, HTTPProvider, PolicyWriter
from .session import ResearchAgent
from .execution import TrackedRunner
from .reporting import report, verify, process_record
from .preparation import prepare
from .discovery import discover
from .. import research as r, stage2
from ..errors import ResearchError
from ..runner import Runner
from ..source_model import DEFAULT_CODE, snapshot


def import_supplied(repetition):
    source = ROOT / 'skills/direct_plate'
    # Existing origin contract includes the original implementation, wrapper and
    # documentation hashes. No fabricated Stage1 discovery artifact is created.
    origin = read(source / 'origin.json')
    from .records import file_sha
    if not origin or origin.get('autonomous_discovery') is not False:
        raise ResearchError('Missing engineering skill provenance')
    for name in ('toolbox.py', 'skill.md', 'source_motion.py'):
        if origin['hashes'][name] != file_sha(source / name):
            raise ResearchError('Supplied skill provenance mismatch: ' + name)
    r._freeze(source, repetition / 'frozen/supplied')
    immutable(repetition / 'supplied_skill.json', {'origin': origin, 'assumed_stage1': True,
              'autonomous_discovery': False})
    snapshot(DEFAULT_CODE, repetition / 'source_models/initial')


def execute(command, run, config, *, provider=None, research_agent=None,
            runner_factory=None, preparation_validator=None, source_only=False):
    run = Path(run).resolve()
    offline = bool(config.get('offline_acceptance'))
    if offline and provider is None and not source_only:
        raise ResearchError('Offline acceptance must use an explicit simulated provider')
    if command == 'resume':
        config = check_resume(run)
        audit = verify(run, reconcile=True)
        if not audit['integrity_ok']:
            raise ResearchError('; '.join(audit['errors']))
        command = 'run'
    cfg = initialize(run, config, offline=offline)
    audit = verify(run, reconcile=True)
    if not audit['integrity_ok']:
        raise ResearchError('; '.join(audit['errors']))
    if command == 'stage2' and cfg['route'] != 'supplied':
        raise ResearchError('stage2 diagnostic requires route=supplied; use adapt for autonomous Stage1')
    gateway = None
    if not source_only:
        if type(cfg.get('model_request_limit')) is not int or cfg['model_request_limit'] < 1:
            raise ResearchError('Explicit --model-request-limit is required for a new formal experiment')
        if provider is None:
            provider = HTTPProvider(cfg.get('model_gateway_url', ''),
                                    cfg.get('model_key_env', 'SIMEX_MODEL_API_KEY'))
        if bool(provider.simulated) != offline:
            raise ResearchError('Provider type must agree with the frozen experiment mode')
        gateway = Gateway(run, cfg['model_request_limit'], provider)
    writer = PolicyWriter(gateway, cfg) if gateway else None
    agent = research_agent or (ResearchAgent(run, gateway, cfg) if gateway else None)
    if agent is not None:
        # Policy decisions cannot accidentally use this persistent research agent.
        if agent is writer:
            raise ResearchError('Research and policy contexts must be separate')
    run.mkdir(parents=True, exist_ok=True)
    def refresh():
        report(run)
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM, terminate)
    owns_run = False
    try:
        with r._lock(run):
            owns_run = True
            if (run / 'STOP').exists():
                if command != 'run':
                    raise ResearchError('Stopped run; use explicit resume')
                (run / 'STOP').unlink()
            save(run / 'process.json', process_record())
            for index in range(cfg['repetitions']):
                repetition = run / 'repetitions' / f'{index:02d}'
                repcfg = read(repetition / 'manifest.json')['config']
                delegate = (runner_factory or Runner)(writer, repetition, repcfg)
                runner = TrackedRunner(delegate, run, repetition, repcfg, refresh=refresh)
                try:
                    stages = ['prepare', 'discover', 'adapt', 'evaluate'] if command == 'run' else [command]
                    if cfg['route'] == 'supplied':
                        import_supplied(repetition)
                        stages = ['source_validation'] + ([] if source_only else (
                            ['adapt', 'evaluate'] if command in ('run', 'stage2') else [command]))
                    for phase in stages:
                        if (run / 'STOP').exists():
                            raise KeyboardInterrupt('Stop requested')
                        save(run / 'state.json', {'state': 'running', 'phase': phase, 'active': f'repetition/{index:02d}'})
                        refresh()
                        if phase == 'prepare':
                            description = Path(repcfg['workspace_description']).read_text()
                            options = {'validator': preparation_validator} if preparation_validator else {}
                            prepare(repetition, repcfg, agent, Path(repcfg['observation']), description,
                                    Path(repcfg['source_assets']), **options)
                        elif phase == 'discover':
                            discover(runner, agent, repetition, repcfg)
                        elif phase == 'source_validation':
                            stage2.source_validation(runner, repetition, repcfg)
                        elif phase == 'adapt':
                            stage2.adapt(runner, agent, repetition, repcfg)
                        elif phase == 'evaluate':
                            # A fresh PolicyWriter has no research session, notes,
                            # images or evaluation-history input. No optimization
                            # agent is invoked after this boundary.
                            writer = PolicyWriter(gateway, repcfg)
                            runner.agent = delegate.agent = writer
                            stage2.evaluate(runner, writer, repetition, repcfg)
                        else:
                            raise ResearchError('Unsupported workspace phase: ' + phase)
                        refresh()
                finally:
                    runner.close()
            save(run / 'state.json', {'state': 'complete', 'phase': 'source_validation' if source_only else command, 'active': None})
    except (Exception, KeyboardInterrupt) as exc:
        if owns_run:
            state = read(run / 'state.json', {})
            save(run / 'state.json', {**state, 'state': 'stopped' if isinstance(exc, KeyboardInterrupt) else 'incomplete',
                                    'reason': str(exc), 'error': type(exc).__name__})
            refresh()
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        if gateway:
            gateway.close()
        if owns_run:
            save(run / 'process.json', {**process_record(), 'active': False})
            refresh()
    return report(run)
