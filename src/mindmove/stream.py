"""Trial-by-trial BCI stream simulator with a safety threshold and a virtual wheelchair.

Synchronous (cue-based) BCI, honest to the dataset: held-out session-E trials
are replayed one at a time; the decoder sees only that trial's EEG.

Decision rule: if max class probability < threshold -> "uncertain, no movement".
WHY: for a wheelchair a wrong move is worse than no move, so we trade a few
missed commands for far fewer wrong ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as C

ACTIONS = {"left_hand": "turn left", "right_hand": "turn right", "feet": "forward", "tongue": "stop"}
HEADINGS = [(0, 1), (1, 0), (0, -1), (-1, 0)]          # N, E, S, W  (dx, dy)
GRID = 10


@dataclass
class Wheelchair:
    x: int = GRID // 2
    y: int = 0
    heading: int = 0                                    # index into HEADINGS (start facing north)
    path: list = field(default_factory=list)

    def __post_init__(self):
        self.path = [(self.x, self.y)]

    def step(self, command: str | None):
        """left/right = turn 90° then move one cell; forward = one cell; stop / None = stay."""
        if command == "turn left":
            self.heading = (self.heading - 1) % 4
        elif command == "turn right":
            self.heading = (self.heading + 1) % 4
        if command in ("turn left", "turn right", "forward"):
            dx, dy = HEADINGS[self.heading]
            self.x = int(np.clip(self.x + dx, 0, GRID - 1))
            self.y = int(np.clip(self.y + dy, 0, GRID - 1))
        self.path.append((self.x, self.y))


@dataclass
class TrialResult:
    index: int
    probs: np.ndarray
    pred: str
    true: str
    confident: bool
    command: str | None

    @property
    def correct(self):
        return self.pred == self.true


def decide(probs, threshold):
    k = int(np.argmax(probs))
    confident = float(probs[k]) >= threshold
    pred = C.CLASSES[k]
    return pred, confident, (ACTIONS[pred] if confident else None)


def run_stream(model, X, y, threshold=0.45, order=None):
    """Generator over TrialResult for every trial in X (one prediction per trial)."""
    order = np.arange(len(y)) if order is None else order
    probs_all = model.predict_proba(X[order])
    for i, p in zip(order, probs_all):
        pred, conf, cmd = decide(p, threshold)
        yield TrialResult(int(i), p, pred, C.CLASSES[int(y[i])], conf, cmd)


def stream_summary(results: list[TrialResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    conf = [r for r in results if r.confident]
    return {
        "n": n,
        "accuracy_all": np.mean([r.correct for r in results]),
        "coverage": len(conf) / n,                                   # share of trials acted on
        "accuracy_when_acting": np.mean([r.correct for r in conf]) if conf else np.nan,
        "wrong_moves": sum(not r.correct for r in conf),
    }
