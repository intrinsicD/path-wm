'''Essential tests for paired_summary.summarize_pairs.

All data is synthetic; no real experiment results are used.
'''

import math

import pytest

from scripts.paired_summary import PairedDataError, summarize_pairs

CASE_A = 'a' * 64
CASE_B = 'b' * 64
MODEL_A = 'c' * 64
MODEL_B = 'd' * 64
CALIB = 'e' * 64


def arm(dataset, seed, projections, successes, newly_solved, cases, initial_successes,
        case_sha, model_sha, step=2000, variant='saved', calib=None):
    return {
        'dataset': dataset, 'seed': seed, 'projections': projections, 'step': step,
        'variant': variant, 'case_sha256': case_sha, 'calibration_rows_sha256': calib,
        'successes': successes, 'cases': cases, 'initial_successes': initial_successes,
        'newly_solved': newly_solved, 'initial_model_sha256': model_sha,
    }


# seed -> (succ@1024, newly@1024, succ@4096, newly@4096)
DS_A = {3072: (30, 12, 34, 15), 3073: (31, 13, 33, 14), 3074: (29, 11, 35, 16)}
DS_B = {3072: (50, 15, 49, 14), 3073: (52, 16, 53, 17), 3074: (48, 14, 55, 20)}


def build_rows():
    rows = []
    for seed, (s0, n0, s1, n1) in DS_A.items():
        rows.append(arm('dsA', seed, 1024, s0, n0, 50, 20, CASE_A, MODEL_A))
        rows.append(arm('dsA', seed, 4096, s1, n1, 50, 20, CASE_A, MODEL_A))
    for seed, (s0, n0, s1, n1) in DS_B.items():
        rows.append(arm('dsB', seed, 1024, s0, n0, 80, 40, CASE_B, MODEL_B))
        rows.append(arm('dsB', seed, 4096, s1, n1, 80, 40, CASE_B, MODEL_B))
    return rows


def test_exact_paired_arithmetic_with_unequal_cases_across_datasets():
    recs = summarize_pairs(build_rows())
    assert [r['dataset'] for r in recs] == ['dsA', 'dsB']
    a, b = recs

    # dsA: 50 cases, 20 initially solved -> 30 initially unsolved.
    assert a['seeds_complete'] == 3 and a['missing_pairs'] == []
    assert [e['delta_successes'] for e in a['per_seed']] == [4, 2, 6]
    assert [e['delta_success_rate_pp'] for e in a['per_seed']] == [8.0, 4.0, 12.0]
    assert [e['delta_unsolved_conditional_pp'] for e in a['per_seed']] == pytest.approx(
        [10.0, 100.0 / 30.0, 500.0 / 30.0])

    st = a['stats']['delta_successes']
    assert (st['n'], st['mean'], st['min'], st['max'], st['range']) == (3, 4.0, 2, 6, 4)
    assert st['sd'] == pytest.approx(2.0)
    assert st['se'] == pytest.approx(2.0 / math.sqrt(3))
    assert a['stats']['delta_success_rate_pp']['mean'] == pytest.approx(8.0)
    assert a['stats']['delta_success_rate_pp']['sd'] == pytest.approx(4.0)
    assert a['stats']['delta_unsolved_conditional_pp']['mean'] == pytest.approx(10.0)
    assert a['stats']['delta_unsolved_conditional_pp']['sd'] == pytest.approx(20.0 / 3.0)

    # dsB: 80 cases -> percentage points use dsB's own denominator, never a pooled one.
    assert [e['delta_successes'] for e in b['per_seed']] == [-1, 1, 7]
    assert [e['delta_success_rate_pp'] for e in b['per_seed']] == [-1.25, 1.25, 8.75]
    assert [e['delta_unsolved_conditional_pp'] for e in b['per_seed']] == pytest.approx(
        [-2.5, 2.5, 15.0])
    assert b['stats']['delta_successes']['mean'] == pytest.approx(7.0 / 3.0)
    assert b['stats']['delta_success_rate_pp']['mean'] == pytest.approx(8.75 / 3.0)
    # datasets stay separate
    assert a['stats']['delta_successes']['n'] == b['stats']['delta_successes']['n'] == 3


def test_mismatched_populations_and_impossible_rows_are_rejected():
    def pair(**over):
        lo = arm('dsA', 3072, 1024, 30, 12, 50, 20, CASE_A, MODEL_A)
        hi = arm('dsA', 3072, 4096, 34, 15, 50, 20, CASE_A, MODEL_A)
        hi.update(over)
        return [lo, hi]

    with pytest.raises(PairedDataError, match='initial_successes differs'):
        summarize_pairs(pair(initial_successes=21))
    with pytest.raises(PairedDataError, match='case_sha256 differs'):
        summarize_pairs(pair(case_sha256=CASE_B))
    with pytest.raises(PairedDataError, match='initial_model_sha256 differs'):
        summarize_pairs(pair(initial_model_sha256=MODEL_B))
    with pytest.raises(PairedDataError, match='cases differs'):
        summarize_pairs(pair(cases=60))

    lo = arm('dsA', 3072, 1024, 30, 12, 50, 20, CASE_A, MODEL_A,
             variant='calibrated', calib=CALIB)
    hi = arm('dsA', 3072, 4096, 34, 15, 50, 20, CASE_A, MODEL_A,
             variant='calibrated', calib='f' * 64)
    with pytest.raises(PairedDataError, match='calibration_rows_sha256 differs'):
        summarize_pairs([lo, hi])

    with pytest.raises(PairedDataError, match='duplicate arm'):
        summarize_pairs(pair() + [arm('dsA', 3072, 4096, 33, 15, 50, 20, CASE_A, MODEL_A)])
    with pytest.raises(PairedDataError, match='successes .* exceeds cases'):
        summarize_pairs(pair(successes=51))
    with pytest.raises(PairedDataError, match='newly_solved .* exceeds initially unsolved'):
        summarize_pairs(pair(newly_solved=31))
    with pytest.raises(PairedDataError, match='unknown seed'):
        summarize_pairs(pair(seed=999))
    with pytest.raises(PairedDataError, match='unknown projections'):
        summarize_pairs(pair(projections=2048))
    with pytest.raises(PairedDataError, match='unknown variant'):
        summarize_pairs(pair(variant='tuned'))
    with pytest.raises(PairedDataError, match='plain non-bool int'):
        summarize_pairs(pair(successes=True))
    with pytest.raises(PairedDataError, match='plain non-bool int'):
        summarize_pairs(pair(successes=float('nan')))
    with pytest.raises(PairedDataError, match='plain non-bool int'):
        summarize_pairs(pair(successes=34.0))
    with pytest.raises(PairedDataError, match='saved rows must carry'):
        summarize_pairs(pair(calibration_rows_sha256=CALIB))


def test_incomplete_pairs_are_explicit_and_never_imputed():
    rows = []
    for seed in (3072, 3073):
        s0, n0, s1, n1 = DS_A[seed]
        rows.append(arm('dsA', seed, 1024, s0, n0, 50, 20, CASE_A, MODEL_A))
        rows.append(arm('dsA', seed, 4096, s1, n1, 50, 20, CASE_A, MODEL_A))
    rows.append(arm('dsA', 3074, 1024, 29, 11, 50, 20, CASE_A, MODEL_A))  # 4096 arm absent
    # calibrated group with a single complete pair
    rows.append(arm('dsA', 3072, 1024, 30, 12, 50, 20, CASE_A, MODEL_A,
                    variant='calibrated', calib=CALIB))
    rows.append(arm('dsA', 3072, 4096, 34, 15, 50, 20, CASE_A, MODEL_A,
                    variant='calibrated', calib=CALIB))

    recs = summarize_pairs(rows)
    cal = next(r for r in recs if r['variant'] == 'calibrated')
    saved = next(r for r in recs if r['variant'] == 'saved')

    assert saved['seeds_complete'] == 2
    assert saved['missing_pairs'] == [{'seed': 3074, 'missing_projections': [4096]}]
    incomplete = saved['per_seed'][2]
    assert incomplete['complete'] is False
    assert 'delta_successes' not in incomplete  # no fabricated zeros
    st = saved['stats']['delta_successes']
    assert st['n'] == 2 and st['mean'] == pytest.approx(3.0)
    assert st['sd'] == pytest.approx(math.sqrt(2.0))
    assert st['se'] == pytest.approx(1.0)

    assert cal['seeds_complete'] == 1
    assert cal['missing_pairs'] == [
        {'seed': 3073, 'missing_projections': [1024, 4096]},
        {'seed': 3074, 'missing_projections': [1024, 4096]},
    ]
    cst = cal['stats']['delta_success_rate_pp']
    assert cst['n'] == 1 and cst['mean'] == pytest.approx(8.0)
    assert cst['sd'] is None and cst['se'] is None
    assert cst['range'] == 0.0


def test_success_components_and_integral_arm_identity_are_validated():
    rows = build_rows()
    rows[0]['newly_solved'] = 0  # 30 successes cannot all come from 20 initial cases.
    with pytest.raises(PairedDataError): summarize_pairs(rows)
    rows = build_rows(); rows[0]['projections'] = 1024.0
    with pytest.raises(PairedDataError): summarize_pairs(rows)
