#!/usr/bin/env python3
"""Exhaustive model of grouped generation's interval-to-slot assignment.

This exercises an independent Python model, NOT Rust code, native inference,
production admission, timing, or memory usage. No third-party dependencies.
Run: python3 scripts/check_grouped_wave_model.py
"""
from itertools import permutations, product


def waves(capacities, intervals):
    slots = sorted(range(len(capacities)), key=lambda s: (capacities[s], s))
    pending = sorted(range(len(intervals)), key=lambda i: (intervals[i][1], intervals[i][0], i))
    result = []
    while pending:
        free = set(slots)
        assigned, rest = [], []
        for row in pending:
            lo, hi = intervals[row]
            slot = next((s for s in slots if s in free and lo <= capacities[s] <= hi), None)
            if slot is None:
                rest.append(row)
            else:
                free.remove(slot)
                assigned.append((row, slot))
        if not assigned:
            return None
        result.append(sorted(assigned))
        pending = rest
    return result


def main():
    cases = 0
    intervals = [(lo, hi) for lo in range(1, 5) for hi in range(lo, 5)]
    for ns in range(1, 4):
        for caps in product(range(1, 5), repeat=ns):
            for nr in range(1, 4):
                for requests in product(intervals, repeat=nr):
                    out = waves(caps, requests)
                    feasible = all(any(lo <= cap <= hi for cap in caps) for lo, hi in requests)
                    assert (out is not None) == feasible
                    if out is not None:
                        assert sorted(i for wave in out for i, _ in wave) == list(range(nr))
                        assert len(out) <= nr
                        for wave in out:
                            assert len({s for _, s in wave}) == len(wave)
                            assert all(requests[i][0] <= caps[s] <= requests[i][1] for i, s in wave)
                        full = nr <= ns and any(
                            all(requests[i][0] <= caps[s] <= requests[i][1] for i, s in enumerate(perm))
                            for perm in permutations(range(ns), nr)
                        )
                        assert (len(out) == 1) == full
                    cases += 1
    assert cases == 93_240
    print(f"PASS: {cases} exhaustive scheduling-model cases (not Rust tests).")


if __name__ == "__main__":
    main()
