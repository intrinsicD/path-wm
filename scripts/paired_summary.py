'''Paired projection-sweep summarizer (1024 vs 4096) for grouped experiment rows.

Authored with Claude via MCP; original contribution and receipt are preserved in
runs/projection_training_2026-09-06/claude. Codex tightened count validation.

Pure standard library. No CLI, no I/O, no plotting, no significance testing,
no pass/fail threshold. Datasets, steps and variants are never pooled, and
cases are never treated as independent training seeds: the only unit of
replication is the training seed, and the only comparison is the within-seed
paired difference between the 1024-projection arm and the 4096-projection arm.
'''

from __future__ import annotations

import math
from statistics import fmean, stdev

DEFAULT_EXPECTED_SEEDS = (3072, 3073, 3074)
BASELINE_PROJECTIONS = 1024
TREATMENT_PROJECTIONS = 4096
VALID_PROJECTIONS = (BASELINE_PROJECTIONS, TREATMENT_PROJECTIONS)
VALID_VARIANTS = ('saved', 'calibrated')

METRICS = (
    'delta_successes',
    'delta_success_rate_pp',
    'delta_unsolved_conditional_pp',
)

# Fields that must be identical between the two arms of one seed pair: the two
# arms must score the same case population, start from the same initial model,
# have the same initial success count, and (for calibrated) the same
# calibration training rows.
PAIR_INVARIANTS = (
    'cases',
    'initial_successes',
    'case_sha256',
    'initial_model_sha256',
    'calibration_rows_sha256',
)

REQUIRED_FIELDS = (
    'dataset', 'seed', 'projections', 'step', 'variant', 'case_sha256',
    'calibration_rows_sha256', 'successes', 'cases', 'initial_successes',
    'newly_solved', 'initial_model_sha256',
)

_HEX = frozenset('0123456789abcdef')


class PairedDataError(ValueError):
    '''Schema, duplicate-arm, impossible-count or within-pair mismatch error.'''


def _count(value, field, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise PairedDataError(
            f'{where}: {field} must be a plain non-bool int; got {value!r} '
            f'(bool, float, NaN and inf are rejected)')
    if value < 0:
        raise PairedDataError(f'{where}: {field} must be >= 0; got {value}')
    return value


def _sha256(value, field, where):
    if not isinstance(value, str):
        raise PairedDataError(
            f'{where}: {field} must be a 64-character sha256 hex digest; got {value!r}')
    text = value.strip().lower()
    if len(text) != 64 or not set(text) <= _HEX:
        raise PairedDataError(
            f'{where}: {field} is not a 64-character sha256 hex digest; got {value!r}')
    return text


def _validate_row(row, index, expected_seeds):
    where = f'row[{index}]'
    if not isinstance(row, dict):
        raise PairedDataError(f'{where}: each row must be a dict; got {type(row).__name__}')
    absent = [f for f in REQUIRED_FIELDS if f not in row]
    if absent:
        raise PairedDataError(f'{where}: missing required fields: {sorted(absent)}')

    dataset = row['dataset']
    if not isinstance(dataset, str) or not dataset.strip():
        raise PairedDataError(f'{where}: dataset must be a non-empty str; got {dataset!r}')
    dataset = dataset.strip()

    seed = row['seed']
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in expected_seeds:
        raise PairedDataError(
            f'{where}: unknown seed {seed!r}; expected one of {list(expected_seeds)}')

    projections = row['projections']
    if isinstance(projections, bool) or not isinstance(projections, int) or projections not in VALID_PROJECTIONS:
        raise PairedDataError(
            f'{where}: unknown projections {projections!r}; expected one of {list(VALID_PROJECTIONS)}')

    variant = row['variant']
    if variant not in VALID_VARIANTS:
        raise PairedDataError(
            f'{where}: unknown variant {variant!r}; expected one of {list(VALID_VARIANTS)}')

    step = _count(row['step'], 'step', where)
    cases = _count(row['cases'], 'cases', where)
    successes = _count(row['successes'], 'successes', where)
    initial_successes = _count(row['initial_successes'], 'initial_successes', where)
    newly_solved = _count(row['newly_solved'], 'newly_solved', where)

    if cases == 0:
        raise PairedDataError(f'{where}: cases must be > 0')
    if successes > cases:
        raise PairedDataError(f'{where}: successes ({successes}) exceeds cases ({cases})')
    if initial_successes > cases:
        raise PairedDataError(
            f'{where}: initial_successes ({initial_successes}) exceeds cases ({cases})')
    unsolved = cases - initial_successes
    if newly_solved > unsolved:
        raise PairedDataError(
            f'{where}: newly_solved ({newly_solved}) exceeds initially unsolved ({unsolved})')
    if newly_solved > successes:
        raise PairedDataError(
            f'{where}: newly_solved ({newly_solved}) exceeds successes ({successes})')

    if successes - newly_solved > initial_successes:
        raise PairedDataError(f'{where}: successes minus newly_solved exceeds initial_successes')

    calibration = row['calibration_rows_sha256']
    if variant == 'saved':
        if calibration is not None:
            raise PairedDataError(
                f'{where}: saved rows must carry calibration_rows_sha256=None; got {calibration!r}')
    else:
        calibration = _sha256(calibration, 'calibration_rows_sha256', where)

    return {
        'dataset': dataset,
        'seed': seed,
        'projections': projections,
        'step': step,
        'variant': variant,
        'cases': cases,
        'successes': successes,
        'initial_successes': initial_successes,
        'newly_solved': newly_solved,
        'case_sha256': _sha256(row['case_sha256'], 'case_sha256', where),
        'initial_model_sha256': _sha256(row['initial_model_sha256'], 'initial_model_sha256', where),
        'calibration_rows_sha256': calibration,
    }


def _check_pair(base, treat, where):
    for field in PAIR_INVARIANTS:
        if base[field] != treat[field]:
            raise PairedDataError(
                f'{where}: {field} differs within the pair '
                f'({BASELINE_PROJECTIONS}={base[field]!r} vs {TREATMENT_PROJECTIONS}={treat[field]!r}); '
                'the two arms are not a comparable paired population')


def _pair_entry(seed, base, treat):
    cases = base['cases']
    initial_successes = base['initial_successes']
    unsolved = cases - initial_successes
    base_rate = 100.0 * base['successes'] / cases
    treat_rate = 100.0 * treat['successes'] / cases
    if unsolved:
        base_cond = 100.0 * base['newly_solved'] / unsolved
        treat_cond = 100.0 * treat['newly_solved'] / unsolved
        delta_cond = treat_cond - base_cond
    else:
        base_cond = treat_cond = delta_cond = None
    return {
        'seed': seed,
        'complete': True,
        'cases': cases,
        'initial_successes': initial_successes,
        'initially_unsolved': unsolved,
        'baseline': {
            'projections': BASELINE_PROJECTIONS,
            'successes': base['successes'],
            'newly_solved': base['newly_solved'],
            'success_rate_pp': base_rate,
            'unsolved_conditional_pp': base_cond,
        },
        'treatment': {
            'projections': TREATMENT_PROJECTIONS,
            'successes': treat['successes'],
            'newly_solved': treat['newly_solved'],
            'success_rate_pp': treat_rate,
            'unsolved_conditional_pp': treat_cond,
        },
        'delta_successes': treat['successes'] - base['successes'],
        'delta_success_rate_pp': treat_rate - base_rate,
        'delta_unsolved_conditional_pp': delta_cond,
    }


def _stats(values):
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': None, 'min': None, 'max': None,
                'range': None, 'sd': None, 'se': None}
    low = min(values)
    high = max(values)
    sd = stdev(values) if n > 1 else None
    return {
        'n': n,
        'mean': fmean(values),
        'min': low,
        'max': high,
        'range': high - low,
        'sd': sd,
        'se': (sd / math.sqrt(n)) if sd is not None else None,
    }


def summarize_pairs(rows, expected_seeds=DEFAULT_EXPECTED_SEEDS):
    '''Summarize 1024-vs-4096 paired arms, grouped by (dataset, step, variant).

    Returns a list of records sorted by (dataset, step, variant). Each record
    holds per-seed paired differences plus mean / min / max / range / sample sd
    (ddof=1) / standard error computed only over complete pairs, together with
    the explicit list of missing pairs. Missing arms are never imputed as zero.
    Raises PairedDataError on any schema, duplicate-arm, impossible-count or
    within-pair population mismatch.
    '''
    seeds = tuple(expected_seeds)
    if not seeds:
        raise PairedDataError('expected_seeds must be non-empty')
    for s in seeds:
        if isinstance(s, bool) or not isinstance(s, int):
            raise PairedDataError(f'expected_seeds must contain plain ints; got {s!r}')
    if len(set(seeds)) != len(seeds):
        raise PairedDataError(f'expected_seeds must be distinct; got {list(seeds)}')

    arms = {}
    for index, row in enumerate(rows):
        record = _validate_row(row, index, seeds)
        key = (record['dataset'], record['step'], record['variant'],
               record['seed'], record['projections'])
        if key in arms:
            raise PairedDataError(
                f'duplicate arm for (dataset, step, variant, seed, projections)={key}')
        arms[key] = record

    out = []
    for dataset, step, variant in sorted({key[:3] for key in arms}):
        per_seed = []
        missing = []
        collected = {m: [] for m in METRICS}
        for seed in seeds:
            base = arms.get((dataset, step, variant, seed, BASELINE_PROJECTIONS))
            treat = arms.get((dataset, step, variant, seed, TREATMENT_PROJECTIONS))
            if base is None or treat is None:
                gone = [p for p, arm in ((BASELINE_PROJECTIONS, base),
                                         (TREATMENT_PROJECTIONS, treat)) if arm is None]
                missing.append({'seed': seed, 'missing_projections': gone})
                per_seed.append({'seed': seed, 'complete': False,
                                 'missing_projections': gone})
                continue
            where = f'{dataset}/step={step}/{variant}/seed={seed}'
            _check_pair(base, treat, where)
            entry = _pair_entry(seed, base, treat)
            for metric in METRICS:
                if entry[metric] is not None:
                    collected[metric].append(entry[metric])
            per_seed.append(entry)
        out.append({
            'dataset': dataset,
            'step': step,
            'variant': variant,
            'baseline_projections': BASELINE_PROJECTIONS,
            'treatment_projections': TREATMENT_PROJECTIONS,
            'expected_seeds': list(seeds),
            'seeds_expected': len(seeds),
            'seeds_complete': sum(1 for e in per_seed if e['complete']),
            'missing_pairs': missing,
            'per_seed': per_seed,
            'stats': {m: _stats(collected[m]) for m in METRICS},
        })
    return out
