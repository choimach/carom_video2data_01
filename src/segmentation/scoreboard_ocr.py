"""
Reading the broadcast scoreboard.

The overlay is the only place the match state is written down, and unlike the
table it stays in the same pixels through every camera cut - which makes it the
one signal that survives the director. It carries what vision cannot supply:
whether a play scored.

Two things make this cheap enough to run over a whole match. The boxes are at
fixed positions, so there is nothing to detect; and their contents change a few
dozen times in two hours, so recognition results are cached against the pixels
they came from and OCR runs only when a box actually changes.

Which player uses which cue ball is read from the board too: each player's row
is drawn in the colour of their ball. That mapping is what connects a score to
the ball seen moving on the table.
"""

import cv2
import numpy as np

# Boxes for the layout used by the 2026 World Cup broadcast, in pixels of a
# 1920x1080 frame: (x, y, w, h). They are scaled to whatever frame arrives - a
# screening preview is half this size, and fixed pixel boxes read the wrong part
# of it, which made a match with a perfectly good scoreboard look like it had
# none at all.
REFERENCE_WIDTH = 1920
DEFAULT_ROIS = {
    "inning": (133, 103, 28, 32),
    "score_top": (400, 101, 34, 34),
    "score_bot": (400, 139, 34, 34),
    "run_top": (444, 101, 40, 34),
    "run_bot": (444, 139, 40, 34),
    "clock": (174, 181, 34, 26),
    "row_top": (210, 101, 150, 34),
    "row_bot": (210, 139, 150, 34),
}
# The shot clock ticks every second and nothing downstream needs it, so it is
# left out: recognising it alone would cost an OCR call per second of match.
DIGIT_FIELDS = ("inning", "score_top", "score_bot", "run_top", "run_bot")
UPSCALE = 6  # the digits are ~30 px tall; OCR is far steadier on an enlarged crop


class Board:
    """One reading of the scoreboard."""

    def __init__(self, inning, scores, runs, clock, rows):
        self.inning = inning
        self.scores = scores  # {"white": int, "yellow": int}
        self.runs = runs  # {"white": int or None, "yellow": int or None}
        self.clock = clock
        self.rows = rows  # {"top": colour, "bot": colour}

    def __repr__(self):
        return (
            f"<Board inn={self.inning} white={self.scores.get('white')}"
            f"(+{self.runs.get('white')}) yellow={self.scores.get('yellow')}"
            f"(+{self.runs.get('yellow')}) clock={self.clock}>"
        )


def _row_colour(patch):
    """Which cue ball this player's row stands for, from the row's background."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # The name sits in the middle of the row, so judge the background from the
    # brightest quarter of the pixels rather than the mean.
    v = hsv[..., 2]
    bright = hsv[v >= np.percentile(v, 75)].reshape(-1, 3)
    hue, sat = float(np.median(bright[:, 0])), float(np.median(bright[:, 1]))
    if sat < 60:
        return "white"
    if 15 <= hue <= 40:
        return "yellow"
    return None


class ScoreboardReader:
    def __init__(self, rois=None, gpu=True, min_confidence=0.30, languages=("en",)):
        self.rois = dict(rois or DEFAULT_ROIS)
        self.min_confidence = min_confidence
        self._reader = None
        self._gpu = gpu
        self._languages = list(languages)
        self._cache = {}
        self.ocr_calls = 0
        self.lookups = 0

    @property
    def reader(self):
        if self._reader is None:
            import easyocr  # imported lazily: loading the models costs ~25 s

            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
        return self._reader

    @staticmethod
    def _fingerprint(patch):
        """Identity of a box's contents, cheap enough to compute every frame.

        Coarse enough that compression noise and a flickering shot clock bar do
        not force a re-read, fine enough that two different digits never land on
        the same key.
        """
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Blur before sampling. Without it the fingerprint tracks the encoder
        # rather than the digits, a new key appears almost every frame, and the
        # cache never hits.
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        # Measured over 695 reads of a 15-minute stretch: a 67% hit rate,
        # against 5% before the blur. The misses that remain are encoder noise
        # on boxes whose digits never changed, so there is more to win here.
        small = cv2.resize(grey, (10, 10), interpolation=cv2.INTER_AREA)
        return (small // 24).tobytes()

    def _digits(self, patch):
        """Recognise the digits in one box, reusing the answer for identical pixels."""
        key = self._fingerprint(patch)
        self.lookups += 1
        if key in self._cache:
            return self._cache[key]

        self.ocr_calls += 1
        h, w = patch.shape[:2]
        big = cv2.resize(patch, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
        grey = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        found = self.reader.readtext(grey, allowlist="0123456789", detail=1)
        text = "".join(t for _, t, c in found if c >= self.min_confidence)
        value = int(text) if text.isdigit() else None
        self._cache[key] = value
        return value

    def _crop(self, frame, name):
        scale = frame.shape[1] / REFERENCE_WIDTH
        x, y, w, h = (int(round(v * scale)) for v in self.rois[name])
        if h < 3 or w < 3 or y + h > frame.shape[0] or x + w > frame.shape[1]:
            return None
        return frame[y:y + h, x:x + w]

    def rows_present(self, frame):
        """Colour of each player row, or None if the overlay is not on screen."""
        rows = {}
        for side in ("top", "bot"):
            patch = self._crop(frame, f"row_{side}")
            if patch is None:
                return None
            colour = _row_colour(patch)
            if colour is None:
                return None
            rows[side] = colour
        # The two rows carry different balls; if they read the same, this is not
        # the scoreboard.
        return rows if rows["top"] != rows["bot"] else None

    def read(self, frame):
        """Read the board, or None when the overlay is not showing."""
        rows = self.rows_present(frame)
        if rows is None:
            return None

        values = {}
        for name in DIGIT_FIELDS:
            patch = self._crop(frame, name)
            values[name] = None if patch is None else self._digits(patch)

        scores, runs = {}, {}
        for side, colour in rows.items():
            scores[colour] = values[f"score_{side}"]
            runs[colour] = values[f"run_{side}"]
        if all(v is None for v in scores.values()):
            return None
        return Board(values["inning"], scores, runs, values.get("clock"), rows)


def read_series(frames, reader=None, step=1):
    """Read a sequence of (frame_index, image) pairs into {frame_index: Board}."""
    reader = reader or ScoreboardReader()
    out = {}
    for position, (index, image) in enumerate(frames):
        if position % step:
            continue
        board = reader.read(image)
        if board is not None:
            out[index] = board
    return out


def running_scores(boards, use_run=True):
    """Per-frame running score for each cue ball, for label_outcomes().

    The blue box holds the score the player came into the inning with and the
    outer box the points made since, so the live total is the sum of the two.
    Reading the blue box alone understates a player in the middle of a run and
    makes their scoring plays look like misses.
    """
    series = {}
    for index, board in boards.items():
        entry = {}
        for colour, score in board.scores.items():
            if score is None:
                continue
            run = board.runs.get(colour) if use_run else None
            entry[colour] = score + (run or 0)
        if entry:
            series[index] = entry
    return series


def turns_from_boards(boards):
    """One entry per player's turn at the table: (start, end, cue_ball, points).

    The board says whose turn it is more plainly than the table does: only the
    player currently shooting has a run box, and that box counts the points
    made in this turn. Taking inning boundaries from here rather than from the
    cue ball seen moving means a turn whose plays were all missed still exists -
    where vision alone silently merged the innings on either side of it.
    """
    frames = sorted(boards)
    turns = []
    for frame in frames:
        board = boards[frame]
        active = [c for c, run in board.runs.items() if run is not None]
        if len(active) != 1:
            continue  # between turns, or the box was unreadable
        colour = active[0]
        run = board.runs[colour] or 0
        if turns and turns[-1][2] == colour and run >= turns[-1][3]:
            # Same turn continuing; a run that goes back down is a new turn.
            turns[-1] = (turns[-1][0], frame, colour, max(turns[-1][3], run))
        else:
            turns.append((frame, frame, colour, run))
    return [tuple(t) for t in turns]
