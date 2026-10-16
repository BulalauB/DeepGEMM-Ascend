"""Analytical routing estimates or exact counts for the current Flash kernel.

Examples:
  python docs/analyze_flash_route_statistics.py --ep 8 --tokens 96 4096
  python docs/analyze_flash_route_statistics.py --ep 8 --counts counts.json

counts.json is a JSON list of one rank's per-expert received route counts
for ONE launch, in local-expert order. Use a delta for cumulative counters.
The analytical model matches the exchangeable iid-score top-k generator in
tests/test_mega_moe.py, not measured DeepSeek model routing or device timing.
"""

import argparse
import json
import math
from pathlib import Path

E, TOPK, H, I, CORES, BLOCK_M = 256, 6, 4096, 2048, 32, 256


def split_expert_rows(rows):
    if rows == 0:
        return []
    blocks = (rows + 255) // 256
    groups = (rows + 15) // 16
    quotient, remainder = divmod(groups, blocks)
    result = []
    begin = 0
    for block in range(blocks):
        capacity = 16 * (quotient + (block < remainder))
        result.append(min(capacity, rows - begin))
        begin += capacity
    return result


def row_metrics(rows):
    blocks = split_expert_rows(rows)
    comm = [sum(max(0, (m + 31 - core) // 32) for m in blocks)
            for core in range(CORES)]
    return {
        'blocks': len(blocks),
        'm128_blocks': sum(m <= 128 for m in blocks),
        'aligned_m_sum': sum((m + 15) // 16 * 16 for m in blocks),
        'comm': comm,
    }


def exact_counts(counts, ep):
    if len(counts) != E // ep or any(type(x) is not int or x < 0 for x in counts):
        raise ValueError('Expected E/EP nonnegative integer counts from one launch')
    valid = [m for count in counts for m in split_expert_rows(count)]
    aligned = [(m + 15) // 16 * 16 for m in valid]
    b = len(valid)
    return {
        'kind': 'exact arithmetic from supplied counts; no timing prediction',
        'received_rows': sum(counts),
        'active_experts': sum(x > 0 for x in counts),
        'routed_blocks': b,
        'tasks_per_layer': 16 * b,
        'waves_per_layer': (b + 1) // 2,
        'tasks_per_core_per_layer': [(b + 1 - c // 16) // 2 for c in range(CORES)],
        'm128_blocks': sum(m <= 128 for m in valid),
        'aligned_row_efficiency': sum(counts) / sum(aligned) if aligned else 1,
        'aligned_rows_per_aic': [sum(aligned[c // 16::2]) for c in range(CORES)],
        'pull_or_push_rows_per_aiv1': [sum(max(0, (m + 31 - c) // 32) for m in valid) for c in range(CORES)],
    }


def statistical_model(ep, tokens):
    import numpy as np
    from scipy.stats import binom

    # Each token chooses six DISTINCT experts uniformly. A given expert is
    # chosen with probability 6/256 independently across the EP*T tokens.
    trials, probability = ep * tokens, TOPK / E
    dist = binom(trials, probability)
    begin, end = max(0, int(dist.ppf(1e-14))), int(dist.ppf(1 - 1e-14))
    support = np.arange(begin, end + 1)
    pmf = dist.pmf(support)
    metrics = [row_metrics(int(x)) for x in support]
    local_experts = E // ep
    expected_blocks = local_experts * sum(float(w) * m['blocks'] for w, m in zip(pmf, metrics))
    expected_aligned = local_experts * sum(float(w) * m['aligned_m_sum'] for w, m in zip(pmf, metrics))
    comm = [local_experts * sum(float(w) * m['comm'][c] for w, m in zip(pmf, metrics)) for c in range(CORES)]
    mean = trials * probability
    # A token's routes to one rank follow a hypergeometric distribution.
    rank_route_variance = trials * TOPK * (1 / ep) * (1 - 1 / ep) * (E - TOPK) / (E - 1)
    return {
        'kind': 'analytical iid-score uniform DISTINCT top-k estimate; NOT measured routing',
        'ep': ep, 'tokens_per_rank': tokens,
        'expert_rows_mean': mean,
        'expert_rows_std': math.sqrt(trials * probability * (1 - probability)),
        'expert_rows_p05_p95': [int(dist.ppf(.05)), int(dist.ppf(.95))],
        'strict_balance_blocks': local_experts * math.ceil(mean / BLOCK_M),
        'expected_blocks': expected_blocks,
        'expected_tasks_per_layer': 16 * expected_blocks,
        'approx_waves_per_layer': expected_blocks / 2,
        'prob_expert_rows_over_768': float(dist.sf(768)),
        'expected_m128_blocks': local_experts * sum(float(w) * m['m128_blocks'] for w, m in zip(pmf, metrics)),
        'expected_aligned_rows': expected_aligned,
        'ratio_of_expected_useful_to_aligned_rows': (TOPK * tokens) / expected_aligned,
        'expected_pull_or_push_rows_per_aiv1': comm,
        'rank_received_rows_mean': TOPK * tokens,
        'rank_received_rows_std': math.sqrt(rank_route_variance),
        'expected_remote_rows_per_rank': TOPK * tokens * (1 - 1 / ep),
        'logical_flops_per_rank_including_shared': 6 * H * I * (TOPK + 1) * tokens,
        'pmf_retained_mass': float(pmf.sum()),
        'notes': 'Expected block totals use linearity and need no independence between experts. Waves are approximately E[B]/2; ceil parity adds <0.5. Aligned efficiency is a ratio of expectations, not a hardware utilization metric.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep', type=int, default=8)
    parser.add_argument('--tokens', type=int, nargs='+', default=[96, 4096])
    parser.add_argument('--counts', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.ep < 1 or E % args.ep or any(t < 1 for t in args.tokens):
        parser.error('EP must divide 256 and tokens must be positive')
    report = exact_counts(json.loads(args.counts.read_text(encoding='utf-8')), args.ep) if args.counts else [statistical_model(args.ep, t) for t in args.tokens]
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(content, encoding='utf-8')
    print(content)


if __name__ == '__main__':
    main()
