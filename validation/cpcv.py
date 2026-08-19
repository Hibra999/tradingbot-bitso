from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPCVFold:
    test_groups: tuple[int, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    episode_segments: tuple[np.ndarray, ...]


class CPCVSplitter:
    def __init__(
        self,
        temporal_groups: int = 6,
        test_groups: int = 2,
        embargo_bars: int = 200,
        max_holding_bars: int = 24,
    ):
        if temporal_groups < 2 or not 0 < test_groups < temporal_groups:
            raise ValueError("CPCV needs at least two groups and fewer test than total groups")
        if embargo_bars < 0 or max_holding_bars < 1:
            raise ValueError("embargo must be non-negative and holding horizon positive")
        self.temporal_groups = temporal_groups
        self.test_group_count = test_groups
        self.embargo_bars = embargo_bars
        self.max_holding_bars = max_holding_bars

    def split(
        self,
        index: pd.DatetimeIndex,
        interval_end: pd.Series | pd.DatetimeIndex,
    ) -> list[CPCVFold]:
        if len(index) < self.temporal_groups or not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("index must be unique, chronological, and at least as long as temporal_groups")
        ends = pd.DatetimeIndex(interval_end)
        if len(ends) != len(index) or bool((ends < index).any()):
            raise ValueError("interval_end must align with index and cannot precede its start")
        end_positions = index.searchsorted(ends, side="left")
        if bool(((end_positions - np.arange(len(index))) > self.max_holding_bars).any()):
            raise ValueError("an interval exceeds max_holding_bars")

        positions = np.arange(len(index))
        groups = tuple(np.asarray(group, dtype=int) for group in np.array_split(positions, self.temporal_groups))
        folds: list[CPCVFold] = []
        for selected in combinations(range(self.temporal_groups), self.test_group_count):
            test = np.concatenate([groups[group] for group in selected])
            train_mask = np.ones(len(index), dtype=bool)
            train_mask[test] = False

            test_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for group_number in selected:
                group = groups[group_number]
                test_ranges.append((index[group[0]], ends[group].max()))
                embargo_end = min(group[-1] + 1 + self.embargo_bars, len(index))
                train_mask[group[-1] + 1 : embargo_end] = False

            candidate = positions[train_mask]
            for start, end in test_ranges:
                overlaps = (index[candidate] <= end) & (ends[candidate] >= start)
                train_mask[candidate[overlaps]] = False

            gaps = np.flatnonzero(np.diff(test) > 1) + 1
            segments = tuple(segment for segment in np.split(test, gaps) if len(segment))
            folds.append(
                CPCVFold(
                    test_groups=selected,
                    train_indices=positions[train_mask],
                    test_indices=test,
                    episode_segments=segments,
                )
            )
        return folds
