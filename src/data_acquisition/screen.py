"""
Deciding whether a match is worth downloading in full.

Only a broadcast shot from above the table can be turned into data: the
calibration needs all four rails in frame, and a ball's position in millimetres
needs a view that is close to orthographic. Plenty of world cup coverage is not
like that - the 2023 Shanghai final is shot from the side, with the table
running off the edge of the picture, and nothing in this pipeline can use it.

Finding that out from the full file costs ten gigabytes and half an hour of
downloading. The 540p rendition is a tenth of the size and answers the question
just as well: whether the table can be calibrated does not depend on
resolution, only on where the camera is. So a match is screened small, and only
the ones that pass are fetched at full quality.
"""

import json
import os
import subprocess
import sys
import tempfile

import cv2

from src.physics.table_calibration import CalibrationError, calibrate

SCREEN_FORMAT = "hls-hd"  # 960x540, about a tenth the size of the original
DEFAULT_SAMPLES = 14


def _yt_dlp():
    """The yt-dlp beside the running interpreter, or whatever is on PATH."""
    beside = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    return beside if os.path.exists(beside) else "yt-dlp"


def probe(url):
    """Title and length, without downloading anything."""
    result = subprocess.run(
        [_yt_dlp(), "--no-warnings", "--dump-single-json", "--skip-download", url],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    if result.returncode:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return {"id": data.get("id"), "title": data.get("title"), "duration": data.get("duration")}


def fetch_preview(url, path, fmt=SCREEN_FORMAT):
    """Download the small rendition. Returns False if it could not be had."""
    result = subprocess.run(
        [_yt_dlp(), "--no-warnings", "--no-part", "-f", fmt, "-o", path, url],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    return result.returncode == 0 and os.path.exists(path)


def assess(video_path, samples=DEFAULT_SAMPLES, skip_fraction=0.25):
    """Can this broadcast be calibrated, and how often?

    Sampling starts a quarter of the way in: these VODs open with anything up to
    an hour of standby card and promotion, and a sample taken there says nothing
    about the camera.
    """
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    if duration <= 0:
        capture.release()
        return {"usable": False, "reason": "unreadable", "calibrated": 0, "samples": 0}

    begin = duration * skip_fraction
    step = (duration - begin) / max(1, samples)
    calibrated, errors, scales, seen = 0, [], [], 0
    for index in range(samples):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int((begin + index * step) * fps))
        ok, frame = capture.read()
        if not ok:
            continue
        seen += 1
        try:
            found = calibrate(frame)
        except CalibrationError:
            continue
        calibrated += 1
        errors.append(found.reprojection_error)
        scales.append(found.mm_per_px)
    capture.release()

    share = calibrated / seen if seen else 0.0
    verdict = {
        "usable": calibrated >= 2,
        "samples": seen,
        "calibrated": calibrated,
        "calibrated_share": share,
        "median_error_mm": sorted(errors)[len(errors) // 2] if errors else None,
        "mm_per_px": sorted(scales)[len(scales) // 2] if scales else None,
    }
    if not verdict["usable"]:
        # The distinction that matters: a camera that never shows the whole
        # table cannot be fixed, while a match that simply has little table time
        # is only poor value.
        verdict["reason"] = "no overhead camera" if calibrated == 0 else "table rarely in frame"
    return verdict


def screen(url, workdir=None, keep_preview=False, verbose=True):
    """Download the small rendition, judge it, and throw it away."""
    workdir = workdir or tempfile.gettempdir()
    os.makedirs(workdir, exist_ok=True)
    details = probe(url) or {}
    name = f"screen_{details.get('id', 'x')}.mp4"
    path = os.path.join(workdir, name)
    try:
        if not fetch_preview(url, path):
            return {"url": url, "usable": False, "reason": "preview download failed", **details}
        verdict = assess(path)
        verdict.update({"url": url, **details})
        if verbose:
            mark = "USE " if verdict["usable"] else "skip"
            extra = (f"calibrated {verdict['calibrated']}/{verdict['samples']}, "
                     f"{verdict['median_error_mm']:.1f} mm"
                     if verdict["usable"] else verdict.get("reason", ""))
            print(f"  {mark} {details.get('title', url)[:52]:52s} {extra}", flush=True)
        return verdict
    finally:
        if not keep_preview and os.path.exists(path):
            os.remove(path)
