"""Calibration and ball detection tested against a synthetic overhead table.

A rendered table is used rather than a video frame so the true scale, corner
positions and ball centres are known exactly and the tests need no media files.
The synthetic geometry copies what was measured on the LIWC 2026 broadcast: the
cushion cloth extends ~19 px past the cushion nose line and the diamonds sit
~35 px past it.
"""

import cv2
import numpy as np
import pytest

from src.physics.ball_detector import BallDetector
from src.physics.table_calibration import (
    TABLE_LENGTH_MM,
    TABLE_WIDTH_MM,
    CalibrationError,
    calibrate,
    detect_cloth_quad,
)
from src.physics.perspective import PerspectiveTransformer

MM_PER_PX = 2.5
NOSE_ORIGIN = (400.0, 270.0)
CLOTH_MARGIN_PX = 19.0
DIAMOND_MARGIN_PX = 35.0

CLOTH_BGR = (235, 150, 45)  # hue ~103, the blue cloth used at this venue
RAIL_BGR = (40, 35, 30)
FLOOR_BGR = (60, 60, 60)
DIAMOND_BGR = (245, 245, 245)
BALL_BGR = {"red": (40, 40, 220), "yellow": (40, 200, 230), "white": (240, 240, 240)}


def _nose_rect():
    x0, y0 = NOSE_ORIGIN
    return x0, y0, TABLE_LENGTH_MM / MM_PER_PX, TABLE_WIDTH_MM / MM_PER_PX


def _to_px(x_mm, y_mm):
    x0, y0 = NOSE_ORIGIN
    return x0 + x_mm / MM_PER_PX, y0 + y_mm / MM_PER_PX


def _draw_diamond(img, x, y, half=7):
    pts = np.array([[x, y - half], [x + half * 0.6, y], [x, y + half], [x - half * 0.6, y]], np.int32)
    cv2.fillPoly(img, [pts], DIAMOND_BGR)


def render_table(balls_mm=None, hide_rails=()):
    """Render an overhead table. balls_mm maps colour -> (x, y) in table mm."""
    img = np.full((1080, 1920, 3), FLOOR_BGR, np.uint8)
    x0, y0, w, h = _nose_rect()

    cv2.rectangle(
        img,
        (int(x0 - DIAMOND_MARGIN_PX - 22), int(y0 - DIAMOND_MARGIN_PX - 22)),
        (int(x0 + w + DIAMOND_MARGIN_PX + 22), int(y0 + h + DIAMOND_MARGIN_PX + 22)),
        RAIL_BGR,
        -1,
    )
    cv2.rectangle(
        img,
        (int(x0 - CLOTH_MARGIN_PX), int(y0 - CLOTH_MARGIN_PX)),
        (int(x0 + w + CLOTH_MARGIN_PX), int(y0 + h + CLOTH_MARGIN_PX)),
        CLOTH_BGR,
        -1,
    )

    for k in range(9):
        x = x0 + k * w / 8.0
        if "top" not in hide_rails:
            _draw_diamond(img, x, y0 - DIAMOND_MARGIN_PX)
        if "bottom" not in hide_rails:
            _draw_diamond(img, x, y0 + h + DIAMOND_MARGIN_PX)
    for j in range(5):
        y = y0 + j * h / 4.0
        if "left" not in hide_rails:
            _draw_diamond(img, x0 - DIAMOND_MARGIN_PX, y)
        if "right" not in hide_rails:
            _draw_diamond(img, x0 + w + DIAMOND_MARGIN_PX, y)

    for colour, (mx, my) in (balls_mm or {}).items():
        px, py = _to_px(mx, my)
        cv2.circle(img, (int(round(px)), int(round(py))), int(round(61.5 / MM_PER_PX / 2)),
                   BALL_BGR[colour], -1)
    return img


BALLS = {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (2400.0, 500.0)}


@pytest.fixture(scope="module")
def frame():
    return render_table(BALLS)


@pytest.fixture(scope="module")
def calibration(frame):
    return calibrate(frame)


def test_all_diamonds_found(calibration):
    assert calibration.diamond_count == 28
    assert {rail: len(v) for rail, v in calibration.diamonds.items()} == {
        "top": 9, "bottom": 9, "left": 5, "right": 5
    }


def test_scale_matches_truth(calibration):
    assert calibration.mm_per_px == pytest.approx(MM_PER_PX, rel=0.01)


def test_reprojection_error_is_sub_millimetre_scale(calibration):
    assert calibration.reprojection_error < 5.0


def test_recovered_corners_match_nose_line(calibration):
    x0, y0, w, h = _nose_rect()
    expected = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], np.float32)
    assert np.abs(calibration.corners - expected).max() < 4.0


def test_rail_offset_is_recovered(calibration):
    expected_mm = DIAMOND_MARGIN_PX * MM_PER_PX
    for offset in calibration.rail_offset_mm:
        assert offset == pytest.approx(expected_mm, abs=6.0)


def test_diamond_index_zero_sits_at_the_starting_corner(calibration):
    """Regression: cv2.fitLine returns an arbitrarily signed direction, which
    once numbered whole rails backwards and still produced a plausible fit."""
    top = {k: (x, y) for k, x, y in calibration.diamonds["top"]}
    assert top[0][0] < top[8][0], "top rail must be numbered left to right"
    left = {k: (x, y) for k, x, y in calibration.diamonds["left"]}
    assert left[0][1] < left[4][1], "left rail must be numbered top to bottom"


def test_nose_corners_map_to_table_origin(calibration):
    x0, y0, w, h = _nose_rect()
    assert np.allclose(calibration.to_table((x0, y0)), (0.0, 0.0), atol=6.0)
    assert np.allclose(
        calibration.to_table((x0 + w, y0 + h)), (TABLE_LENGTH_MM, TABLE_WIDTH_MM), atol=6.0
    )


def test_calibration_survives_one_hidden_rail():
    """A player leaning over a rail should cost that rail, not the frame."""
    calibration = calibrate(render_table(BALLS, hide_rails=("left",)))
    assert "left" not in calibration.diamonds, "an unusable rail is dropped, not guessed at"
    assert calibration.diamond_count == 23
    assert calibration.mm_per_px == pytest.approx(MM_PER_PX, rel=0.01)
    # The offset for the hidden pair falls back to the visible pair's value.
    assert calibration.rail_offset_mm[1] == pytest.approx(DIAMOND_MARGIN_PX * MM_PER_PX, abs=6.0)


def test_frame_without_a_table_is_rejected():
    noise = np.full((1080, 1920, 3), FLOOR_BGR, np.uint8)
    assert detect_cloth_quad(noise) is None
    with pytest.raises(CalibrationError):
        calibrate(noise)


def test_cloth_outline_overstates_the_table(calibration, frame):
    """The reason calibration uses diamonds: the cloth edge is not the table."""
    quad = detect_cloth_quad(frame)
    cloth_length_px = np.hypot(*(quad[1] - quad[0]))
    nose_length_px = TABLE_LENGTH_MM / MM_PER_PX
    assert cloth_length_px > nose_length_px + 2 * CLOTH_MARGIN_PX - 3


def test_detector_finds_all_three_balls_at_the_right_place(frame, calibration):
    detections = BallDetector(calibration).detect(frame)
    assert set(detections) == {"red", "yellow", "white"}
    for colour, (mx, my) in BALLS.items():
        got = detections[colour]["mm"]
        assert np.allclose(got, (mx, my), atol=12.0), f"{colour}: {got} != {(mx, my)}"


def test_detector_reports_missing_balls_rather_than_guessing(calibration):
    frame = render_table({"white": BALLS["white"]})
    detections = BallDetector(calibration).detect(frame)
    assert set(detections) == {"white"}


def test_detected_ball_size_matches_calibration(frame, calibration):
    detections = BallDetector(calibration).detect(frame)
    expected_radius = (61.5 / calibration.mm_per_px) / 2.0
    for det in detections.values():
        assert det["radius_px"] == pytest.approx(expected_radius, rel=0.25)


def test_perspective_transformer_accepts_a_calibration(calibration):
    transformer = PerspectiveTransformer.from_calibration(calibration)
    x0, y0, _, _ = _nose_rect()
    assert np.allclose(transformer.transform_point(x0, y0), (0.0, 0.0), atol=6.0)


@pytest.mark.parametrize("width,height", [(1920, 1080), (1280, 720), (960, 540)])
def test_calibration_survives_a_downscaled_frame(frame, width, height):
    """Screening a match cheaply means judging it at 540p, so the detection
    sizes cannot be fixed pixel counts measured at 1920.

    They were, and it cost ten matches: a diamond covers some forty pixels at
    full size and eight at half, which falls through the area floor, and a 3x3
    opening erases what survives. Ten perfectly good broadcasts were screened
    out as having no overhead camera at all.
    """
    small = cv2.resize(frame, (width, height))
    calibration = calibrate(small)
    assert calibration.diamond_count == 28
    assert calibration.mm_per_px == pytest.approx(MM_PER_PX * 1920 / width, rel=0.02)


def test_ball_size_scales_with_the_frame_too():
    positions = {"white": (700.0, 400.0), "yellow": (1900.0, 1000.0), "red": (2400.0, 500.0)}
    small = cv2.resize(render_table(positions), (960, 540))
    calibration = calibrate(small)
    found = BallDetector(calibration).detect(small)
    assert set(found) == {"red", "yellow", "white"}
    for colour, (x, y) in positions.items():
        assert np.allclose(found[colour]["mm"], (x, y), atol=25.0)
