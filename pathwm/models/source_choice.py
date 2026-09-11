"""Outcome-only action values for two opaque evidence sources."""

import math


class SourceChoice:
    def __init__(self, window=None, change_block=None, change_z=None):
        if window is not None and (type(window) is not int or window < 1):
            raise ValueError("Window must be a positive integer")
        if change_block is not None and (
            type(change_block) is not int or change_block < 1 or window is not None
        ):
            raise ValueError("Change block must be positive and exclude windowing")
        if change_z is not None and (
            not math.isfinite(change_z)
            or change_z <= 0
            or change_block is None
            or change_block < 2
        ):
            raise ValueError("Positive finite change multiplier requires block>=2")
        self.change_z = change_z
        self.squares = [0.0, 0.0]
        self.change_block = change_block
        self.pending = [[], []]
        self.resets = [0, 0]
        self.window = window
        self.history = [[], []]
        self.counts = [0, 0]
        self.sums = [0.0, 0.0]

    def observe(self, source, gain):
        if type(source) is not int or source not in (0, 1) or not math.isfinite(gain):
            raise ValueError("Invalid source feedback")
        self.squares[source] += float(gain) ** 2
        self.counts[source] += 1
        self.sums[source] += float(gain)
        if self.window is not None:
            self.history[source].append(float(gain))
            if len(self.history[source]) > self.window:
                self.history[source].pop(0)
            self.counts[source] = len(self.history[source])
            self.sums[source] = sum(self.history[source])

        if self.change_block is not None:
            block = self.pending[source]
            block.append(float(gain))
            if len(block) == self.change_block:
                older_count = self.counts[source] - len(block)
                recent_sum = sum(block)
                if older_count >= self.change_block:
                    older_mean = (self.sums[source] - recent_sum) / older_count
                    threshold = 0.15
                    recent_squares = sum(g * g for g in block)
                    if self.change_z is not None:
                        older_variance = max(
                            0.0,
                            (
                                self.squares[source]
                                - recent_squares
                                - older_count * older_mean**2
                            )
                            / (older_count - 1),
                        )
                        recent_variance = max(
                            0.0,
                            (recent_squares - recent_sum**2 / len(block))
                            / (len(block) - 1),
                        )
                        threshold = max(
                            threshold,
                            self.change_z
                            * math.sqrt(
                                older_variance / older_count
                                + recent_variance / len(block)
                                + 0.0001
                            ),
                        )
                    if abs(recent_sum / len(block) - older_mean) >= threshold:
                        self.counts[source] = len(block)
                        self.sums[source] = recent_sum
                        self.squares[source] = recent_squares
                        self.resets[source] += 1
                self.pending[source] = []

    def choose(self):
        means = [s / n if n else 0.0 for s, n in zip(self.sums, self.counts)]
        if max(means) <= 0 or means[0] == means[1]:
            return None
        return max(range(2), key=means.__getitem__)

    def snapshot(self):
        state = dict(counts=self.counts.copy(), sums=self.sums.copy())
        if self.window is not None:
            state.update(window=self.window, history=[h.copy() for h in self.history])
        if self.change_block is not None:
            state.update(
                change_block=self.change_block,
                threshold=0.15,
                pending=[h.copy() for h in self.pending],
                resets=self.resets.copy(),
            )
        if self.change_z is not None:
            state.update(change_z=self.change_z, squares=self.squares.copy())
        return state
