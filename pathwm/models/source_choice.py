"""Outcome-only action values for two opaque evidence sources."""

import math


class SourceChoice:
    def __init__(self):
        self.counts = [0, 0]
        self.sums = [0.0, 0.0]

    def observe(self, source, gain):
        if type(source) is not int or source not in (0, 1) or not math.isfinite(gain):
            raise ValueError("Invalid source feedback")
        self.counts[source] += 1
        self.sums[source] += float(gain)

    def choose(self):
        means = [s / n if n else 0.0 for s, n in zip(self.sums, self.counts)]
        if max(means) <= 0 or means[0] == means[1]:
            return None
        return max(range(2), key=means.__getitem__)

    def snapshot(self):
        return dict(counts=self.counts.copy(), sums=self.sums.copy())
