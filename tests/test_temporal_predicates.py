"""Sustained elevation distinguishes a final observed window from a height peak."""
import copy

import pytest

from simex.scoring import score
from simex.tasks import make_task, source_task_from_spec, validate_predicate


def case():
    task = make_task('far_plate', 0)
    task['predicate'] = {'kind': 'sustained_lift', 'object': 'green', 'min': .05, 'seconds': .5}
    obj = {'position': [.55, .17, .083], 'initial_height': .003, 'max_height': .15}
    metrics = {'sim_time': 2.1, 'objects': {'green': obj}, 'tote': task['scene']['tote']}
    samples = [{'sim_time': t, 'objects': {'green': copy.deepcopy(obj)}}
               for t in [1., 1.25, 1.5, 1.75, 2.]]
    return task, metrics, samples


def test_window_can_end_between_trace_samples_and_use_a_preceding_boundary():
    task, metrics, samples = case()
    validate_predicate(task['predicate'], ['green'])
    assert score(task, metrics, samples=samples)
    task['predicate']['seconds'] = .6
    assert score(task, metrics, samples=samples)
    assert not score(task, metrics)  # A final point or historical peak is insufficient.


def test_transient_peak_keeps_legacy_lift_score_but_not_sustained_score():
    task, metrics, samples = case()
    for sample in samples[3:]:
        sample['objects']['green']['position'][2] = .003
    metrics['objects']['green']['position'][2] = .003
    assert not score(task, metrics, samples=samples)
    task['predicate'] = {'kind': 'lift', 'object': 'green', 'min': .05}
    assert score(task, metrics)
    assert score(task, metrics, samples=samples)


@pytest.mark.parametrize('change', [
    'final_drop', 'interior_drop', 'missing_position', 'missing_object', 'nan_height',
    'gap', 'stale', 'short_window', 'duplicate_time', 'reversed_time', 'future_time',
    'nan_time', 'missing_time', 'missing_initial', 'invalid_final_position', 'no_samples',
])
def test_incomplete_or_lowered_measurements_fail_closed(change):
    task, metrics, samples = case()
    if change == 'final_drop':
        metrics['objects']['green']['position'][2] = .003
    elif change == 'interior_drop':
        samples[3]['objects']['green']['position'][2] = .003
    elif change == 'missing_position':
        samples[3]['objects']['green'].pop('position')
    elif change == 'missing_object':
        samples[3]['objects'].clear()
    elif change == 'nan_height':
        samples[3]['objects']['green']['position'][2] = float('nan')
    elif change == 'gap':
        samples.pop(3)
    elif change == 'stale':
        metrics['sim_time'] = 2.5
    elif change == 'short_window':
        samples = samples[3:]
    elif change == 'duplicate_time':
        samples[3]['sim_time'] = samples[2]['sim_time']
    elif change == 'reversed_time':
        samples[3], samples[4] = samples[4], samples[3]
    elif change == 'future_time':
        samples[-1]['sim_time'] = 2.2
    elif change == 'nan_time':
        samples[3]['sim_time'] = float('nan')
    elif change == 'missing_time':
        metrics.pop('sim_time')
    elif change == 'missing_initial':
        metrics['objects']['green'].pop('initial_height')
    elif change == 'invalid_final_position':
        metrics['objects']['green']['position'] = [0, 0]
    elif change == 'no_samples':
        samples = []
    assert not score(task, metrics, samples=samples)


def test_earlier_lower_samples_do_not_invalidate_a_complete_elevated_final_window():
    task, metrics, samples = case()
    samples[0]['objects']['green']['position'][2] = .003
    assert score(task, metrics, samples=samples)


def test_sustained_elevation_does_not_claim_attachment_or_stationarity():
    task, metrics, samples = case()
    for index, sample in enumerate(samples):
        sample['objects']['green']['position'][0] += .03 * index
    # Horizontal motion is allowed: this predicate tests elevation only.
    assert score(task, metrics, samples=samples)


@pytest.mark.parametrize('fields', [
    {'seconds': .49}, {'seconds': 5.01}, {'seconds': True}, {'seconds': float('nan')},
    {'seconds': float('inf')}, {'min': 0}, {'min': True}, {'min': float('nan')},
    {'unknown': 1},
])
def test_generated_temporal_predicate_is_bounded(fields):
    predicate = {**case()[0]['predicate'], **fields}
    with pytest.raises(ValueError):
        source_task_from_spec({'predicate': predicate})


def test_window_is_required_only_for_the_new_predicate():
    predicate = case()[0]['predicate']
    generated = source_task_from_spec({'predicate': predicate})
    assert generated['predicate'] == predicate
    with pytest.raises(ValueError):
        validate_predicate({k: v for k, v in predicate.items() if k != 'seconds'}, ['green'])
    with pytest.raises(ValueError):
        validate_predicate({**predicate, 'kind': 'lift'}, ['green'])
