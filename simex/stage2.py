"""Stage2 experiment with an explicitly supplied skill and independent source code models."""
from __future__ import annotations

import copy
import ast
import csv
import difflib
import hashlib
import json
from pathlib import Path

from . import research as r
from .errors import ResearchError
from .source_model import DEFAULT_CODE, MODEL_HELP, SourceModelError, check_model, model_hash, snapshot
from .tasks import make_task, with_placement_contract

PROTOCOL = 'supplied_skill_stage2_v1'
METHODS = ('skill_transfer', 'skill_stage2')
DEFAULTS = {'protocol': PROTOCOL, 'stage1_toolbox': 'simex/skills/direct_plate',
            'placement_contract': 'released_placement_v1', 'camera_resolution': [640, 480],
            'source_calibration_profile': 'code_v1', 'adapt_trials': 5, 'repairs': 4,
            'screen_tasks': 5, 'align_attempts': 3, 'eval_trials': 5, 'seed': 20261023,
            'max_agent_calls': 2000, 'model': 'gpt-6-astra', 'reasoning_effort': 'medium',
            'stage2_reasoning_effort': 'medium', 'evaluation_reasoning_effort': 'medium'}
SOURCE_CASES = [('far_plate', 42)] + [('far_plate', seed) for seed in range(9, 14)] + [
    ('far_to_near', 42), ('shifted_single', 42)]
VALIDATION_SCRIPT = "print('SUPPLIED_SKILL_RESULT', tb.pick_place_plate())\n"


def camera_task(task, cfg):
    task = copy.deepcopy(task)
    task['scene']['cameras'].update(zip(('width', 'height'), cfg['camera_resolution']))
    return task


def check_library(path):
    r._check_frozen(path)
    if r._json(Path(path) / 'frozen.json')['sha256'] != r.toolbox_hash(path):
        raise ResearchError(f'Frozen capability hash mismatch: {path}')


def implementation_hashes():
    root = Path(__file__).parent
    files = ['stage2.py', 'source_model.py', 'source_model_process.py', 'source_geometry.py',
             'backends/mujoco_backend.py', 'runner.py', 'worker.py', 'scoring.py', 'release.py',
             'research.py', 'research_context.py', 'tasks.py', 'policy_process.py', 'perception.py',
             'agent.py', 'contracts.py', 'wire.py', 'cli.py', 'evidence.py', 'temporal.py']
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def source_asset_hashes():
    root = Path(__file__).resolve().parents[1] / 'assets/panda'
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*')) if path.is_file()}


def initialize(run, config):
    run = Path(run).resolve()
    cfg = {**DEFAULTS, **config}
    cfg['stage1_toolbox'] = str(Path(cfg['stage1_toolbox']).resolve())
    if (cfg['protocol'] != PROTOCOL or cfg['source_calibration_profile'] != 'code_v1'
            or cfg['placement_contract'] != 'released_placement_v1' or cfg.get('stage1_parent')):
        raise ResearchError('Supplied-skill Stage2 requires code_v1, released placement and no Stage1 parent')
    if cfg['camera_resolution'] != [640, 480]:
        raise ResearchError('The supplied direct_plate experiment fixes cameras at 640x480')
    if (cfg['model'] != 'gpt-6-astra' or any(cfg[k] != 'medium' for k in
            ('reasoning_effort', 'stage2_reasoning_effort', 'evaluation_reasoning_effort'))):
        raise ResearchError('This protocol fixes GPT-6 Astra and medium reasoning for both methods')
    for key in ('adapt_trials', 'repairs', 'screen_tasks', 'align_attempts', 'eval_trials', 'max_agent_calls', 'seed'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key in ('align_attempts', 'seed') else 1):
            raise ResearchError(f'Invalid integer setting: {key}')
    fixed = {'adapt_trials': 5, 'repairs': 4, 'screen_tasks': 5, 'align_attempts': 3,
             'eval_trials': 5, 'max_agent_calls': 2000}
    if any(cfg[k] != value for k, value in fixed.items()):
        raise ResearchError('This frozen protocol requires 5 adaptation, 4 repairs, 5 screens, '
                            '3 model edits, 5 evaluations and an independent 2000-call cap')
    bundle = Path(cfg['stage1_toolbox'])
    r._validate_toolbox(r._contents(bundle))
    origin = r._json(bundle / 'origin.json')
    if not origin or origin.get('autonomous_discovery') is not False:
        raise ResearchError('A supplied engineering-skill origin record is required')
    for name in ('toolbox.py', 'skill.md', 'source_motion.py'):
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != origin['hashes'][name]:
            raise ResearchError(f'Supplied skill origin hash mismatch: {name}')
    old = r._json(run / 'manifest.json')
    if old:
        if (old.get('format_version') != 4 or old['config'] != cfg or old['implementation'] != implementation_hashes()
                or old['provenance'] != r._provenance(cfg) or old['source_assets'] != source_asset_hashes()):
            raise ResearchError('Stage2 protocol, source implementation or settings changed; use a new run')
        if old['supplied_skill_sha256'] != r.toolbox_hash(bundle):
            raise ResearchError('Supplied capability changed')
        check_library(run / 'frozen/supplied')
        check_model(run / 'source_models/initial')
        ledger = r._json(run / 'target_ledger.json')
        if not ledger or _schedule_hash(ledger['slots']) != old['target_schedule_sha256']:
            raise ResearchError('Frozen target schedule changed')
        from .release import CONTRACT_SPEC
        if old['placement_contract_spec'] != CONTRACT_SPEC:
            raise ResearchError('Frozen placement contract changed')
        audit = verify(run)
        if not audit['integrity_ok']:
            raise ResearchError('Stage2 artifact integrity failure: ' + '; '.join(audit['errors']))
        return cfg
    run.mkdir(parents=True, exist_ok=True)
    r._freeze(bundle, run / 'frozen/supplied')
    for name in ('source_motion.py', 'origin.json'):
        (run / 'frozen/supplied' / name).write_bytes((bundle / name).read_bytes())
    snapshot(DEFAULT_CODE, run / 'source_models/initial')
    slots = {}
    for i, task in enumerate(r.TASK_IDS):
        for j in range(cfg['adapt_trials']):
            slots[f'adapt/{task}/{j:02d}'] = {'phase': 'adapt', 'task_id': task, 'trial': j,
                'seed': cfg['seed']+20000+i*1000+j, 'state': 'pending'}
        for method in METHODS:
            for j in range(cfg['eval_trials']):
                slots[f'evaluate/{method}/{task}/{j:02d}'] = {'phase': 'evaluate', 'method': method,
                    'task_id': task, 'trial': j, 'seed': cfg['seed']+30000+i*1000+j, 'state': 'pending'}
    from .release import CONTRACT_SPEC
    r._save(run / 'target_ledger.json', {'slots': slots})
    r._save(run / 'supplied_skill.json', {'origin': origin, 'assumed_stage1': True,
        'autonomous_discovery': False, 'inherited_model_calls': 0, 'toolbox_sha256': r.toolbox_hash(bundle)})
    r._save(run / 'manifest.json', {'format_version': 4, 'config': cfg, 'methods': list(METHODS),
        'provenance': r._provenance(cfg), 'implementation': implementation_hashes(),
        'source_assets': source_asset_hashes(),
        'supplied_skill_sha256': r.toolbox_hash(bundle), 'target_schedule_sha256': _schedule_hash(slots),
        'initial_source_model_sha256': model_hash(run / 'source_models/initial'),
        'placement_contract_spec': CONTRACT_SPEC})
    return cfg


def _schedule_hash(slots):
    return r._digest({key: {k: v for k, v in slot.items() if k in ('phase', 'method', 'task_id', 'trial', 'seed')}
                      for key, slot in slots.items()})


def _history_payload(history):
    # Keep prior decisions visible; the complete record is paged through the
    # evidence broker instead of repeating hundreds of old screening outcomes.
    keys = ('trial', 'target_success', 'target_status', 'selected', 'reason',
            'toolbox_sha256', 'source_model_sha256')
    return {'task_history': [{**{k: x[k] for k in keys},
                              'alignment_status': x['alignment']['status']} for x in history],
            'history_archive': json.dumps(history, ensure_ascii=False)}


def seal_rollout(out):
    out = Path(out)
    names = ('research_request.json', 'runner_request.json', 'policy.py', 'policy_generation.json',
             'result.json', 'execution.json', 'public_evidence.json', 'policy_stdout.log',
             'private/scene.json', 'private/metrics.json', 'private/object_trace.jsonl')
    if (out / 'operation_input.json').exists():
        names += ('operation_input.json', 'unverified_result.json', 'executed_library/toolbox.py',
                  'executed_library/skill.md', 'executed_library/frozen.json')
    paths = [out / name for name in names if (out / name).is_file()]
    paths += [p for p in (out / 'public').rglob('*') if p.is_file()]
    r._save(out / 'integrity.json', {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths})


def check_rollout(out):
    out = Path(out)
    record = r._json(out / 'integrity.json')
    if not record or not {'result.json', 'research_request.json'} <= record.keys():
        raise ResearchError(f'Missing rollout integrity record: {out}')
    for name, digest in record.items():
        path = out / name
        if not path.resolve().is_relative_to(out.resolve()) or not path.is_file():
            raise ResearchError(f'Missing rollout artifact: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ResearchError(f'Rollout artifact hash mismatch: {path}')


def source_validation(runner, run, cfg):
    library, model = run / 'frozen/supplied', run / 'source_models/initial'
    check_library(library)
    check_model(model)
    results = []
    for i, (task_id, seed) in enumerate(SOURCE_CASES):
        task = camera_task(with_placement_contract(make_task(task_id, seed), cfg['placement_contract']), cfg)
        result = r._run(runner, 'mujoco', task, library, run / 'source_validation' / f'{i:03d}',
                        VALIDATION_SCRIPT, source_model=model)
        results.append({'task': task_id, 'seed': seed, 'success': bool(result.get('success')),
                        'status': result['status'], 'output_dir': result['output_dir'],
                        'actual_started': result.get('actual_started') is True})
    record = {'complete': len(results) == len(SOURCE_CASES),
              'passed': all(x['success'] and x['actual_started'] for x in results),
              'successes': sum(x['success'] for x in results), 'scheduled': len(SOURCE_CASES),
              'toolbox_sha256': r.toolbox_hash(library), 'source_model_sha256': model_hash(model),
              'scope': 'Supplied-skill integration regression, not Stage1 discovery or independent generalization.',
              'records': results}
    r._save(run / 'source_validation.json', record)
    if not record['passed']:
        raise ResearchError('Supplied skill source validation failed; target adaptation has not been authorized by this gate')
    return record


def _alignment(runner, agent, run, directory, cfg, task, library, script, target, model, history):
    from .backends.mujoco_backend import MujocoBackend
    # Build only the public reconstruction's nominal source XML, without
    # starting physics or inspecting any target-engine state or assets.
    builder = MujocoBackend()
    from .tasks import validate_scene
    source_scene = validate_scene(task['scene'], calibrated=True)
    builder.width, builder.height = cfg['camera_resolution']
    baseline_xml = builder._scene_xml(source_scene)
    directory.mkdir(parents=True, exist_ok=True)
    xml_path = directory / 'baseline.xml'
    if xml_path.exists() and xml_path.read_text() != baseline_xml:
        raise ResearchError('Fixed source reconstruction XML changed')
    xml_path.write_text(baseline_xml)
    source = r._run(runner, 'mujoco', task, library, directory / 'initial', script, source_model=model)
    entries = []
    aligned = False
    edits = 0
    for index in range(cfg['align_attempts'] + 1):
        last = index == cfg['align_attempts']
        instruction = ('Compare the target and source fixed-policy executions. Judge outcome-relevant '
            'behavior, not scores alone. Return {"aligned":bool,"reason":str,"model_code":str, '
            '"matched_observations":[{"phase":str,"target_observation":str,"source_observation":str}], '
            '"remaining_mismatch":str}. A confirmed alignment requires concrete visual/action observations '
            'of corresponding phases. Both executions failing is insufficient. '
            'When not aligned, propose complete source model code only. The script and toolbox are frozen. '
            'Explain the phase, observed discrepancy and remaining uncertainty. ' + MODEL_HELP)
        if last:
            instruction += '\nThis is the final assessment. No edits remain; omit model_code.'
        decision = r._ask(agent, directory / f'assessment_{index:02d}.json', 'align', instruction,
            {'task': task['description'], 'script': script, 'toolbox': r._contents(library),
             'source_model_code': (model / 'model.py').read_text(), 'source_model_sha256': model_hash(model),
             'source_scene': task['scene'], 'baseline_source_xml': baseline_xml,
             'target': r._public(target), 'source': r._public(source), 'alignment_history': entries,
             **_history_payload(history), 'remaining_edits': cfg['align_attempts'] - index})
        entry = {'assessment': index, 'assessed_model_sha256': model_hash(model),
                 'decision': decision, 'source': r._outcome_summary(r._public(source))}
        entries.append(entry)
        matches = decision.get('matched_observations')
        supported = (target.get('actual_started') is True and source.get('actual_started') is True
                     and source['status'] not in ('invalid_source_model', 'invalid_policy', 'generation_error')
                     and isinstance(matches, list) and bool(matches) and all(isinstance(x, dict) and
                         all(isinstance(x.get(k), str) and x[k].strip() for k in
                             ('phase', 'target_observation', 'source_observation')) for x in matches)
                     and isinstance(decision.get('reason'), str) and bool(decision['reason'].strip())
                     and isinstance(decision.get('remaining_mismatch'), str))
        entry['alignment_evidence_complete'] = bool(supported)
        if decision.get('aligned') is True and supported:
            aligned = True
            break
        if last:
            break
        edits += 1
        candidate_dir = directory / f'attempt_{index:02d}'
        try:
            code = decision.get('model_code')
            candidate = snapshot(code, candidate_dir / 'model')
            (candidate_dir / 'model.diff').write_text(''.join(difflib.unified_diff(
                (model / 'model.py').read_text().splitlines(True), code.splitlines(True),
                fromfile='previous/model.py', tofile='candidate/model.py')))
            replay = r._run(runner, 'mujoco', task, library, candidate_dir / 'replay', script,
                            source_model=candidate)
            if replay['status'] == 'invalid_source_model':
                raise SourceModelError(replay.get('reason', 'Invalid source model execution'))
            model, source = candidate, replay
            validation = {'valid': True, 'source_model_sha256': model_hash(model)}
        except (SourceModelError, SyntaxError) as exc:
            validation = {'valid': False, 'reason': str(exc), 'retained_model_sha256': model_hash(model)}
        entry['edit_validation'] = validation
        r._save(candidate_dir / 'validation.json', validation)
    result = {'status': 'agent_confirmed_aligned' if aligned else 'update_limit_reached_unconfirmed',
              'updates_attempted': edits, 'final_assessment_present': True,
              'source_model': str(model), 'source_model_sha256': model_hash(model), 'history': entries}
    r._save(directory / 'history.json', result)
    return model, source, result


def _screen_and_select(runner, agent, directory, cfg, task_id, task, library, model,
                       script, target, source, alignment, history, seeds):
    common = {'task': task['description'], 'toolbox': r._contents(library), 'script': script,
              'source_model_code': (model / 'model.py').read_text(),
              'source_model_sha256': model_hash(model), 'target': r._public(target),
              'source': r._public(source) if source else None, 'alignment': alignment, **_history_payload(history)}
    baseline = directory / 'alignment/baseline.xml'
    if baseline.exists():
        common['baseline_source_xml'] = baseline.read_text()
    proposal = r._ask(agent, directory / 'hypotheses.json', 'repair_hypotheses',
        f'Return {{"hypotheses":[str,...]}} with exactly {cfg["repairs"]} distinct, testable reusable '
        'capability repair hypotheses. Address observed failures, preserve uncertainty about source mismatch. '
        'Source model code is read-only at this stage.', common)
    ideas = proposal.get('hypotheses')
    if (not isinstance(ideas, list) or len(ideas) != cfg['repairs']
            or any(not isinstance(x, str) or not x.strip() for x in ideas) or len(set(ideas)) != len(ideas)):
        raise ResearchError('Expected distinct nonempty repair hypotheses')
    candidates = {'current': library}
    details = {}
    for i, hypothesis in enumerate(ideas):
        name = f'candidate_{i:02d}'
        out = directory / name
        answer = r._ask(agent, out / 'proposal.json', 'repair',
            'Implement only this hypothesis. Return complete {"toolbox.py":str,"skill.md":str}. '
            'Preserve APIs used by the executed script. Modify only the capability library, never the model.\n'
            + r.TOOLBOX_CONTRACT, {**common, 'hypothesis': hypothesis})
        candidate, validation = r._candidate(answer, out)
        details[name] = {'hypothesis': hypothesis, 'validation': validation, 'changes': r._changes(library, answer)}
        if candidate is not None:
            candidates[name] = candidate
    proposal = r._ask(agent, directory / 'screen_proposal.json', 'screen_tasks',
        f'Generate exactly {cfg["screen_tasks"]} source generalization scenarios for this same task. '
        'Return {"tasks":[...]}. All candidates share these scenes and the fixed source model. '
        'Preserve the required success predicate exactly; vary placements and plausible physical conditions. '
        'Never use hidden target parameters or seeds. ' + r.TASK_SPEC_HELP,
        {'task_id': task_id, 'description': task['description'], 'source_scene_template': task['scene'],
         'required_predicate': task['predicate'], 'suggested_source_seeds': seeds,
         'source_model_code': (model / 'model.py').read_text(), 'hypotheses': ideas})
    tasks = [camera_task(t, cfg) for t in r._tasks_from_response(proposal, cfg['screen_tasks'])]
    if any(t['predicate'] != task['predicate'] for t in tasks):
        raise ResearchError('Screening scenarios changed the frozen predicate')
    r._record_scene_variations(proposal, tasks, directory / 'screen_scene_variations.json')
    r._save(directory / 'screen_tasks.json', tasks)
    r._save(directory / 'screen_contract.json', {'source_model_sha256': model_hash(model),
        'tasks_sha256': r._digest(tasks), 'fixed_task_sha256': r._digest(task),
        'fixed_script_sha256': r._digest(script),
        'libraries': {name: r.toolbox_hash(path) for name, path in candidates.items()}})
    evidence = {}
    for name, candidate in candidates.items():
        out = directory / 'screen' / name
        replay = r._run(runner, 'mujoco', task, candidate, out / 'replay', script,
                        source_model=model) if script is not None else None
        results = r._suite(runner, tasks, candidate, out / 'generalization', source_model=model)
        evidence[name] = {'fixed_replay': r._public(replay) if replay else None,
                          'generalization': [r._public(x) for x in results]}
    selection = r._ask(agent, directory / 'repair_selection.json', 'repair_select',
        'Choose a candidate or retain current. Return {"selected":str,"reason":str}. '
        'Use videos, traces and scores as evidence, not an automatic score ranking. '
        'State which tests are trustworthy given remaining model mismatch. A local correction requires '
        'evidence of its predicted effect, remaining failure phases and regressions. More motion alone '
        'does not prove improvement. Failed full tasks remain failures.',
        {**common, 'hypotheses': details, 'available_choices': list(candidates), 'screen': evidence,
         'source_task': task, 'generalization_tasks': tasks,
         'toolboxes': {k: r._contents(v) for k, v in candidates.items()}})
    chosen = selection.get('selected')
    if cfg.get('protocol') == 'simex_workspace_v1':
        if chosen not in candidates or not isinstance(selection.get('reason'), str) or not selection['reason'].strip():
            raise ResearchError('Selection requires a tested candidate and an explicit agent rationale')
        selected_evidence = evidence[chosen]
        # A batch that only produced invalid policies is not a physical test of
        # a new library. Retaining the already accepted parent is always allowed.
        if chosen != 'current' and not any(x and x['status'] not in ('invalid_policy', 'generation_error', 'invalid_source_model')
                for x in [selected_evidence['fixed_replay'], *selected_evidence['generalization']]):
            raise ResearchError('Untested candidate cannot become the frozen capability')
    if chosen not in candidates:
        selection = {**selection, 'requested': chosen, 'selected': 'current',
                     'validation_error': 'Unknown or invalid candidate; retained current'}
        chosen = 'current'
    return candidates[chosen], selection, details, evidence


def adapt(runner, agent, run, config):
    run = Path(run).resolve()
    with r._lock(run):
        workspace = config.get('protocol') == 'simex_workspace_v1'
        cfg = r._json(run / 'manifest.json')['config'] if workspace else initialize(run, config)
        initial_library = run / ('frozen/stage1' if workspace and cfg['route'] == 'learned' else 'frozen/supplied')
        if workspace and cfg['route'] == 'learned':
            if not r._json(run / 'discovery.json'):
                raise ResearchError('Freeze and select all Stage1 members before adaptation')
        else:
            check = r._json(run / 'source_validation.json', {})
            if not check.get('passed') or check.get('toolbox_sha256') != r.toolbox_hash(initial_library):
                raise ResearchError('All supplied-skill source acceptance cases must pass first')
        if callable(getattr(agent, 'set_phase', None)):
            agent.set_phase('stage2')
        if workspace:
            runner.agent.set_phase('stage2')
        done = r._json(run / 'adaptation.json')
        if done:
            for task in r.TASK_IDS:
                check_library(run / 'frozen/adapted' / task)
                check_model(Path(done['tasks'][task]['source_model']))
            return done
        outcomes = {}
        for task_index, task_id in enumerate(r.TASK_IDS):
            library = initial_library
            model = run / 'source_models/initial'
            history = []
            for trial in range(cfg['adapt_trials']):
                directory = run / 'adaptation' / task_id / f'trial_{trial:02d}'
                check_library(library)
                r._freeze(library, directory / 'executed_toolbox')
                if workspace:
                    runner.target_source_context = model
                r._progress(run, 'adaptation_trial_started', task=task_id, trial=trial+1)
                target = r._target(runner, run, f'adapt/{task_id}/{trial:02d}', library, directory / 'target')
                script = r._executed_script(target)
                task = r._json(directory / 'source_reconstruction.json')
                if task is None:
                    if workspace:
                        from .lab.preparation import reconstruction
                        task, assumptions = reconstruction(task_id, r._public(target)['evidence'].get('initial_visual_estimates', {}))
                        r._save(directory / 'reconstruction_provenance.json', assumptions)
                    else:
                        task = camera_task(with_placement_contract(r._source_reconstruction(
                            task_id, r._public(target)['evidence']), cfg['placement_contract']), cfg)
                    r._save(directory / 'source_reconstruction.json', task)
                initial_hash, model_before = r.toolbox_hash(library), model_hash(model)
                source = None
                alignment = {'status': 'fixed_policy_unavailable', 'history': [],
                             'source_model_sha256': model_hash(model)}
                if script is not None:
                    model, source, alignment = _alignment(runner, agent, run, directory / 'alignment',
                        cfg, task, library, script, target, model, history)
                library, selection, candidates, evidence = _screen_and_select(runner, agent, directory,
                    cfg, task_id, task, library, model, script, target, source, alignment, history,
                    [cfg['seed']+40000+task_index*1000+trial*100+j for j in range(cfg['screen_tasks'])])
                frozen = r._freeze(library, directory / 'selected_toolbox')
                for name in ('toolbox.py', 'skill.md'):
                    (directory / (name + '.diff')).write_text(''.join(difflib.unified_diff(
                        (directory / 'executed_toolbox' / name).read_text().splitlines(True),
                        (frozen / name).read_text().splitlines(True), fromfile='executed/' + name,
                        tofile='selected/' + name)))
                library = frozen
                summary = {'trial': trial, 'target_success': bool(target.get('success')),
                    'target_status': target['status'], 'target': r._outcome_summary(r._public(target)),
                    'selected': selection['selected'], 'reason': selection.get('reason'),
                    'toolbox_before_sha256': initial_hash, 'toolbox_sha256': r.toolbox_hash(library),
                    'source_model_before_sha256': model_before, 'source_model_sha256': model_hash(model),
                    'source_model': str(model), 'fixed_script_sha256': r._digest(script),
                    'alignment': alignment, 'candidates': candidates,
                    'screen': {k: {'fixed_replay': r._outcome_summary(v['fixed_replay']),
                        'generalization': [r._outcome_summary(x) for x in v['generalization']]}
                        for k, v in evidence.items()}}
                r._save(directory / 'summary.json', summary)
                history.append(summary)
                r._progress(run, 'adaptation_trial_complete', task=task_id, trial=trial+1,
                            target_status=target['status'], selected=selection['selected'])
            final = r._freeze(library, run / 'frozen/adapted' / task_id)
            final_model = snapshot((model / 'model.py').read_text(), run / 'source_models/adapted' / task_id)
            outcomes[task_id] = {'toolbox': str(final), 'source_model': str(final_model), 'trials': history}
        result = {'tasks': outcomes}
        r._save(run / 'adaptation.json', result)
        return result


def evaluate(runner, agent, run, config):
    run = Path(run).resolve()
    with r._lock(run):
        workspace = config.get('protocol') == 'simex_workspace_v1'
        manifest = r._json(run / 'manifest.json') if workspace else None
        cfg = manifest['config'] if workspace else initialize(run, config)
        if not r._json(run / 'adaptation.json'):
            raise ResearchError('Finish all adaptation before frozen evaluation')
        if callable(getattr(agent, 'set_phase', None)):
            agent.set_phase('evaluation')
        if workspace:
            runner.agent.set_phase('evaluation')
        records = []
        for task in r.TASK_IDS:
            for trial in range(cfg['eval_trials']):
                for method in manifest['methods'] if workspace else METHODS:
                    libraries = {'direct_cap': 'frozen/initial', 'stage1_transfer': 'frozen/stage1',
                                 'skill_transfer': 'frozen/supplied', 'simex': f'frozen/adapted/{task}',
                                 'skill_stage2': f'frozen/adapted/{task}'}
                    library = run / libraries[method]
                    check_library(library)
                    if workspace:
                        runner.target_source_context = run / ('source_models/adapted/' + task
                            if method in ('simex', 'skill_stage2') else 'source_models/initial')
                    result = r._target(runner, run, f'evaluate/{method}/{task}/{trial:02d}', library,
                        run / 'evaluation' / method / task / f'trial_{trial:02d}')
                    records.append({'method': method, 'task_id': task, 'trial': trial,
                                    'success': bool(result.get('success')), 'status': result['status'],
                                    'output_dir': result['output_dir'], 'toolbox_sha256': r.toolbox_hash(library)})
        result = {'records': records}
        r._save(run / 'evaluation.json', result)
        return result


def report(run):
    run = Path(run).resolve()
    manifest = r._json(run / 'manifest.json')
    slots = list(r._json(run / 'target_ledger.json')['slots'].values())
    rows = []
    for task in r.TASK_IDS:
        for method in METHODS:
            selected = [x for x in slots if x['phase'] == 'evaluate' and x['task_id'] == task and x['method'] == method]
            done = [x for x in selected if x['state'] == 'complete']
            wins = sum(bool(x.get('success')) for x in done)
            rows.append({'task_id': task, 'method': method, 'completed': len(done), 'scheduled': len(selected),
                         'successes': wins, 'success_rate': wins/len(selected) if len(done) == len(selected) and selected else None})
    complete = all(x['state'] == 'complete' for x in slots)
    calls = list((run / 'agent_calls').glob('*/request.json'))
    used = sum(r._json(p.parent / 'result.json', {}).get('budget_charged') is not False for p in calls)
    source = r._json(run / 'source_validation.json', {})
    trials = [r._json(p) for p in sorted((run / 'adaptation').glob('*/trial_*/summary.json'))]
    improvement = {}
    for task in r.TASK_IDS:
        a, b = [next(x for x in rows if x['task_id'] == task and x['method'] == method) for method in METHODS]
        improvement[task] = {'observed_improvement': b['successes'] > a['successes'] if complete else None,
                            'baseline_ceiling': a['successes'] == a['scheduled'] if complete else None}
    execution_rows = []
    for key, slot in r._json(run / 'target_ledger.json')['slots'].items():
        output = Path(slot['output_dir']) if slot.get('output_dir') else None
        execution_rows.append({'slot': key, 'phase': slot['phase'], 'method': slot.get('method', 'adaptation'),
            'task_id': slot['task_id'], 'trial': slot['trial'] + 1, 'state': slot['state'],
            'success': slot.get('success'), **(_execution_diagnostic(output, run) if output else {})})
    result = {'protocol': PROTOCOL, 'complete': complete, 'target_completed': sum(x['state'] == 'complete' for x in slots),
        'target_scheduled': len(slots), 'supplied_skill': source, 'stage1_discovery_performed': False,
        'adaptation_complete': (run / 'adaptation.json').exists(), 'evaluation': rows,
        'target_ability_observed': any(x['successes'] for x in rows if x['method'] == 'skill_stage2'),
        'adaptation_target_successes': sum(bool(x.get('success')) for x in slots if x['phase'] == 'adapt'),
        'last_error': r._json(run / 'last_error.json'),
        'improvement': improvement, 'agent_budget_used': used, 'agent_budget_inherited': 0,
        'target_executions': execution_rows, 'adaptation_trace': trials,
        'limitations': ['One independent research seed; no statistical significance claim',
            'Supplied engineering skill; autonomous Stage1 not evaluated',
            'Source XML keeps topology/assets/constraints fixed; numeric geometry and control code are editable']}
    r._save(run / 'report.json', result)
    with (run / 'target_executions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in execution_rows for k in row)))
        writer.writeheader()
        writer.writerows(execution_rows)
    text = ['# 预置 direct_plate 技能的 Stage2 实验', '',
        '能力来自人工提供的 direct_plate，假定为 Stage1 输出；本运行没有执行或验证自主 Stage1。',
        f"源域接入验收：{source.get('successes', 0)}/{source.get('scheduled', len(SOURCE_CASES))}；"
        f"目标槽位完成：{result['target_completed']}/{len(slots)}。",
        '评分为 released_placement_v1；流程完成、任务成功与适应改善分别报告。', '',
        '| 方法 | 任务 | 成功/计划 | 已完成 | 成功率 |', '|---|---|---:|---:|---:|']
    for row in rows:
        rate = '未完成' if row['success_rate'] is None else f"{row['success_rate']:.1%}"
        text.append(f"| {row['method']} | {row['task_id']} | {row['successes']}/{row['scheduled']} | {row['completed']} | {rate} |")
    text += ['', 'skill_transfer：预置技能直接迁移；skill_stage2：经过五轮目标反馈及源域筛选后的冻结能力库。',
             f'模型预算：{used}/{manifest["config"]["max_agent_calls"]}，没有继承旧调用或旧目标反馈。',
             '单次研究运行不宣称统计显著或复现论文成功率。自动判据不证明运输全程持续夹持，需结合视频与阶段记录。', '',
             '| 任务/轮次 | 目标成功 | 对齐 | 选择 | 模型版本 | 能力库版本 | 证据 |', '|---|---|---|---|---|---|---|']
    for task in r.TASK_IDS:
        for path in sorted((run / 'adaptation' / task).glob('trial_*/summary.json')):
            x = r._json(path)
            text.append(f"| {task}/{x['trial']+1} | {x['target_success']} | {x['alignment']['status']} | {x['selected']} | "
                        f"{x['source_model_sha256'][:12]} | {x['toolbox_sha256'][:12]} | "
                        f"[逐轮记录]({path.relative_to(run)}) / [能力差异]({path.parent.relative_to(run)}/toolbox.py.diff) / "
                        f"[校准]({path.parent.relative_to(run)}/alignment/history.json) |")
    text += ['', '阶段诊断只表示公开执行记录到达该阶段；保持和运输开始不等于全程抓稳。'
        '完整成功仍取独立 released_placement_v1 评分。缺少阶段日志时记为 unavailable。', '',
        '| 目标槽位 | 状态 | 最后公开阶段 | 保持开始/运输开始 | 评分细项 | 视频/证据 |',
        '|---|---|---|---|---|---|']
    for row in execution_rows:
        if not row.get('output_dir'):
            continue
        videos = ' / '.join(f'[视频{n+1}]({p})' for n, p in enumerate(row['videos'])) or '无视频'
        text.append(f"| {row['slot']} | {row.get('status', row['state'])} | {row['last_public_phase']} | "
            f"{row['hold_started']}/{row['transport_started']} | {row['predicate_checks']} | "
            f"{videos} / [结果]({row['output_dir']}/result.json) |")
    text += ['', '源域接入验收视频（固定脚本，不计入模型生成策略的目标评估）：']
    for item in source.get('records', []):
        output = Path(item['output_dir'])
        path = output.relative_to(run)
        text.append(f"- {item['task']} seed={item['seed']}：{item['status']}；"
                    f"[结果]({path}/result.json)，[视频]({path}/public/overhead.mp4)")
    for task, value in improvement.items():
        if value['baseline_ceiling']:
            text.append(f'\n{task}：直接迁移已全部成功，本设置无法测出进一步成功率增益。')
        elif complete:
            text.append(f"\n{task}：本轮{'观察到成功次数增加' if value['observed_improvement'] else '未观察到成功率改善'}。")
    if result['last_error']:
        text.append('\n最近停止/错误记录：' + json.dumps(result['last_error'], ensure_ascii=False))
    (run / 'report.md').write_text('\n'.join(text)+'\n')
    return result


def _execution_diagnostic(out, run):
    result = r._json(out / 'result.json', {})
    events = []
    stdout = out / 'policy_stdout.log'
    for line in stdout.read_text().splitlines() if stdout.exists() else []:
        if line.startswith('PLATE_PHASE '):
            try:
                item = ast.parse(line[len('PLATE_PHASE '):], mode='eval').body
                # Observations may contain NumPy repr calls. Read only the
                # literal phase value; never evaluate the observation payload.
                if isinstance(item, ast.Dict):
                    for key, value in zip(item.keys, item.values):
                        if isinstance(key, ast.Constant) and key.value == 'phase':
                            phase = ast.literal_eval(value)
                            if isinstance(phase, str):
                                events.append(phase)
            except (SyntaxError, ValueError, TypeError):
                pass
    checks = {}
    metrics = r._json(out / 'private/metrics.json')
    request = r._json(out / 'research_request.json', {})
    if metrics and request.get('task'):
        from .scoring import score
        task = request['task']
        for predicate in task['predicate'].get('all', [task['predicate']]):
            try:
                checks[predicate['object'] + ':' + predicate['kind']] = score({**task, 'predicate': predicate}, metrics)
            except (KeyError, TypeError, ValueError):
                checks[str(predicate)] = None
    return {'output_dir': str(out.relative_to(run)), 'status': result.get('status', 'incomplete'),
            'reason': result.get('reason', ''), 'last_public_phase': events[-1] if events else 'unavailable',
            'hold_started': 'hold' in events, 'transport_started': 'transport' in events,
            'predicate_checks': checks,
            'release_detail': (metrics or {}).get('release_evidence', {}).get('green', {}),
            'videos': [str(p.relative_to(run)) for p in sorted((out / 'public').glob('*.mp4'))]}


def verify(run):
    run = Path(run).resolve()
    errors = []
    checked = 0
    try:
        manifest = r._json(run / 'manifest.json')
        ledger = r._json(run / 'target_ledger.json')['slots']
        if _schedule_hash(ledger) != manifest['target_schedule_sha256']:
            errors.append('Frozen target schedule changed')
        if (run / 'discovery.json').exists():
            errors.append('Supplied skill must not fabricate a Stage1 discovery record')
        for p in (run / 'frozen').rglob('frozen.json'):
            check_library(p.parent)
        for p in (run / 'source_models').rglob('frozen.json'):
            check_model(p.parent)
        for p in (run / 'adaptation').glob('*/trial_*/alignment/attempt_*/model/frozen.json'):
            check_model(p.parent)
        if model_hash(run / 'source_models/initial') != manifest['initial_source_model_sha256']:
            errors.append('Initial source model changed')
        if r.toolbox_hash(run / 'frozen/supplied') != manifest['supplied_skill_sha256']:
            errors.append('Supplied library changed')
        for p in (run / 'adaptation').glob('*/trial_*/*/frozen.json'):
            check_library(p.parent)
        for p in run.rglob('integrity.json'):
            check_rollout(p.parent)
        for key, slot in ledger.items():
            if slot['state'] != 'complete':
                continue
            out = Path(slot['output_dir'])
            check_rollout(out)
            result = r._json(out / 'result.json')
            if result is None or r._digest(result) != slot['result_sha256']:
                errors.append(f'{key}: missing or changed result')
            request = r._json(out / 'research_request.json', {})
            if request.get('toolbox_sha256') != slot['toolbox_sha256']:
                errors.append(f'{key}: executed capability hash mismatch')
            if request.get('source_model_sha256') or request.get('source_config'):
                errors.append(f'{key}: source calibration applied to target')
        for trial in (run / 'adaptation').glob('*/trial_*'):
            library = trial / 'executed_toolbox'
            if library.exists():
                for path in (trial / 'alignment').rglob('research_request.json'):
                    if r._json(path)['toolbox_sha256'] != r.toolbox_hash(library):
                        errors.append(f'{path}: calibration changed the fixed library')
            contract = r._json(trial / 'screen_contract.json')
            if contract:
                tasks = r._json(trial / 'screen_tasks.json')
                if r._digest(tasks) != contract['tasks_sha256']:
                    errors.append(f'{trial}: frozen screening scenes changed')
                for path in (trial / 'screen').rglob('research_request.json'):
                    request = r._json(path)
                    name = path.relative_to(trial / 'screen').parts[0]
                    expected_task = (r._json(trial / 'source_reconstruction.json') if path.parent.name == 'replay'
                                     else tasks[int(path.parent.name)])
                    if (request['source_model_sha256'] != contract['source_model_sha256']
                            or request['toolbox_sha256'] != contract['libraries'][name]
                            or request['task'] != expected_task):
                        errors.append(f'{path}: screening inputs differ from common frozen contract')
            script = trial / 'target/policy.py'
            replays = list((trial / 'alignment').rglob('replay/policy.py')) + list((trial / 'screen').glob('*/replay/policy.py'))
            if (trial / 'alignment/initial/policy.py').exists():
                replays.append(trial / 'alignment/initial/policy.py')
            for replay in replays:
                if not script.exists() or replay.read_bytes() != script.read_bytes():
                    errors.append(f'{replay}: fixed script changed')
                checked += 1
        validation = r._json(run / 'source_validation.json')
        if validation:
            for record in validation['records']:
                out = Path(record['output_dir'])
                check_rollout(out)
                result = r._json(out / 'result.json')
                if any(record[k] != result.get(k) for k in ('success', 'status', 'actual_started')):
                    errors.append(f'{out}: source validation summary changed')
    except (OSError, KeyError, ValueError, ResearchError) as exc:
        errors.append(str(exc))
        ledger = locals().get('ledger', {})
    completed = sum(x['state'] == 'complete' for x in ledger.values())
    return {'integrity_ok': not errors, 'hash_verification': True,
            'complete': bool(ledger) and completed == len(ledger),
            'completed_target_trials': completed, 'scheduled_target_trials': len(ledger),
            'fixed_replays_checked': checked, 'errors': errors}
