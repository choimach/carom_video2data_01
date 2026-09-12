"""Scoreboard reading tested on a rendered overlay.

Everything here avoids the OCR engine itself: the digit recogniser is stubbed,
so what is under test is the part that can silently be wrong - which row means
which cue ball, what counts as the overlay being on screen, and how the two
score boxes combine into a running total.
"""

import cv2
import numpy as np
import pytest

from src.segmentation.scoreboard_ocr import (
    DEFAULT_ROIS,
    Board,
    ScoreboardReader,
    _row_colour,
    running_scores,
)

WHITE_ROW_BGR = (250, 251, 249)
YELLOW_ROW_BGR = (40, 146, 200)


def render_board(top_bgr=WHITE_ROW_BGR, bot_bgr=YELLOW_ROW_BGR):
    """A frame carrying just enough of the overlay for the reader to work on."""
    frame = np.full((1080, 1920, 3), (60, 60, 60), np.uint8)
    for name, (x, y, w, h) in DEFAULT_ROIS.items():
        colour = (200, 80, 20)  # the blue boxes
        if name == "row_top":
            colour = top_bgr
        elif name == "row_bot":
            colour = bot_bgr
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, -1)
    # A name written across each row, as on the real board.
    cv2.putText(frame, "P.V. BAO", (215, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, "M. ZANETTI", (215, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


class StubReader(ScoreboardReader):
    """A reader whose digit recognition is supplied rather than inferred."""

    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self._values = values
        self._order = []

    def _digits(self, patch):
        self.lookups += 1
        name = self._order.pop(0)
        return self._values.get(name)

    def read(self, frame):
        self._order = ["inning", "score_top", "score_bot", "run_top", "run_bot"]
        return super().read(frame)


def test_row_colour_identifies_each_players_cue_ball():
    frame = render_board()
    x, y, w, h = DEFAULT_ROIS["row_top"]
    assert _row_colour(frame[y:y + h, x:x + w]) == "white"
    x, y, w, h = DEFAULT_ROIS["row_bot"]
    assert _row_colour(frame[y:y + h, x:x + w]) == "yellow"


def test_scores_are_keyed_by_cue_ball_not_by_row():
    """The board is the only place the player-to-ball mapping is written."""
    values = {"inning": 7, "score_top": 7, "score_bot": 6, "run_top": 6, "run_bot": None}
    board = StubReader(values).read(render_board())
    assert board.rows == {"top": "white", "bot": "yellow"}
    assert board.scores == {"white": 7, "yellow": 6}
    assert board.runs == {"white": 6, "yellow": None}

    # Swap the rows: the same digits must now belong to the other ball.
    swapped = StubReader(values).read(render_board(YELLOW_ROW_BGR, WHITE_ROW_BGR))
    assert swapped.scores == {"yellow": 7, "white": 6}


def test_overlay_absent_reads_as_nothing():
    plain = np.full((1080, 1920, 3), (60, 60, 60), np.uint8)
    assert ScoreboardReader().rows_present(plain) is None
    assert StubReader({}).read(plain) is None


def test_two_rows_of_the_same_colour_are_not_a_scoreboard():
    both_white = render_board(WHITE_ROW_BGR, WHITE_ROW_BGR)
    assert ScoreboardReader().rows_present(both_white) is None


def test_running_total_adds_the_run_to_the_confirmed_score():
    """The blue box is the score coming into the inning; the outer box is the
    points made since. A player mid-run is understated by the blue box alone."""
    boards = {
        100: Board(6, {"white": 5, "yellow": 5}, {"white": 2, "yellow": None}, None, {}),
        200: Board(6, {"white": 7, "yellow": 5}, {"white": None, "yellow": 0}, None, {}),
    }
    series = running_scores(boards)
    assert series[100]["white"] == 7, "5 confirmed + 2 in the current run"
    assert series[200]["white"] == 7, "the run folded into the blue box; no jump"
    assert series[100]["yellow"] == 5


def test_reading_the_blue_box_alone_makes_a_run_look_like_misses():
    """Regression on the analysis, not the code: scoring 12 points was read as
    7 because the in-progress run was ignored."""
    boards = {100: Board(7, {"white": 7}, {"white": 6}, None, {})}
    assert running_scores(boards, use_run=True)[100]["white"] == 13
    assert running_scores(boards, use_run=False)[100]["white"] == 7


def test_fingerprint_ignores_noise_but_not_a_changed_digit():
    """What decides whether OCR runs again: too sensitive and it runs on every
    frame, too coarse and a score change goes unnoticed."""
    reader = ScoreboardReader()
    frame = render_board()
    x, y, w, h = DEFAULT_ROIS["score_top"]
    patch = frame[y:y + h, x:x + w]
    assert reader._fingerprint(patch) == reader._fingerprint(patch.copy())
    altered = patch.copy()
    cv2.putText(altered, "8", (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    assert reader._fingerprint(patch) != reader._fingerprint(altered)

    rng = np.random.default_rng(0)
    noisy = np.clip(patch.astype(int) + rng.normal(0, 3, patch.shape), 0, 255).astype(np.uint8)
    assert reader._fingerprint(patch) == reader._fingerprint(noisy), "encoder noise must not force a re-read"
