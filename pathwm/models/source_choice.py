"""Outcome-only action values for two opaque evidence sources."""

import math


class SourceChoice:
    def __init__(self, window=None):
        if window is not None and (type(window) is not int or window < 1):
            raise ValueError("Window must be a positive integer")
        self.window = window
        self.history = [[], []]
        self.counts = [0, 0]
        self.sums = [0.0, 0.0]

    def observe(self, source, gain):
        if type(source) is not int or source not in (0, 1) or not math.isfinite(gain):
            raise ValueError("Invalid source feedback")
        self.counts[source] += 1
        self.sums[source] += float(gain)
        if self.window is not None:
            self.history[source].append(float(gain))
            if len(self.history[source]) > self.window:
                self.history[source].pop(0)
            self.counts[source] = len(self.history[source])
            self.sums[source] = sum(self.history[source])

    def choose(self):
        means = [s / n if n else 0.0 for s, n in zip(self.sums, self.counts)]
        if max(means) <= 0 or means[0] == means[1]:
            return None
        return max(range(2), key=means.__getitem__)

    def snapshot(self):
        state = dict(counts=self.counts.copy(), sums=self.sums.copy())
        if self.window is not None:
            state.update(window=self.window, history=[h.copy() for h in self.history])
        return state
