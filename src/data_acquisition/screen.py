"""
Deciding whether a match is worth downloading in full.

Only a broadcast shot from above the table can be turned into data: the
calibration needs all four rails in frame, and a ball's position in millimetres
needs a view that is close to orthographic. Plenty of world cup coverage is not
like that - the 2023 Shanghai final is shot from the side, with the table
running off the edge of the picture, and nothing in this pipeline can use it.

Answering that from the full file costs ten gigabytes and half an hour. It does
not have to: these VODs are served as HLS, a playlist of six-second segments
that can be fetched one at a time. Fourteen segments scattered across a match
is eighty megabytes and half a minute, and they come at the broadcast's own
resolution, so the screen sees exactly what the scan will see.

The earlier approach - downloading the whole 540p rendition instead - was both
slower and wrong. Shrinking a broadcast by half leaves the camera angle intact
but not the diamonds: the Ankara final is shot from further back than the SOOP
world cup coverage, and its markers are three pixels across at 540p. They were
not found, and a perfectly good overhead broadcast was written off as having no
overhead camera at all. Screening at full size removes the question.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import cv2

from src.physics.table_calibration import (CalibrationError, calibrate,
                                           detect_cloth_quad, looks_like_a_table,
                                           table_scale)

STREAM_FORMAT = "hls-original"  # the broadcast's own resolution
SCREEN_FORMAT = "hls-hd"  # 960x540; only for the whole-preview fallback
DEFAULT_SAMPLES = 30
DEEP_SAMPLES = 70  # when the first pass could not decide


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


def playlist_url(url, fmt=STREAM_FORMAT):
    """Direct HLS playlist for one rendition, or None if it cannot be resolved."""
    result = subprocess.run(
        [_yt_dlp(), "--no-warnings", "-f", fmt, "-g", url],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines[-1].strip() if result.returncode == 0 and lines else None


def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def segment_index(playlist):
    """Parse a media playlist into (base url, init bytes, [(start, name)], total).

    Start times come from adding up the EXTINF durations rather than from any
    timestamp in the stream, which is what makes a segment addressable by the
    second it plays at.
    """
    text = _get(playlist).decode("utf-8", "replace")
    base = playlist.rsplit("/", 1)[0]
    init_name, entries, clock, duration = None, [], 0.0, 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-MAP:"):
            found = re.search(r'URI="([^"]+)"', line)
            init_name = found.group(1) if found else None
        elif line.startswith("#EXTINF:"):
            duration = float(line[len("#EXTINF:"):].rstrip(","))
        elif line and not line.startswith("#"):
            entries.append((clock, line if line.startswith("http") else base + "/" + line))
            clock += duration
    # Fragmented MP4 segments are undecodable without the initialisation
    # segment that carries the codec headers; plain MPEG-TS has none.
    init = _get(base + "/" + init_name) if init_name else b""
    return base, init, entries, clock


def sample_frames(url, samples=DEFAULT_SAMPLES, skip_fraction=0.25, fmt=STREAM_FORMAT,
                  workdir=None, exclude=()):
    """One frame from each of `samples` segments spread across the match.

    Sampling starts a quarter of the way in: these VODs open with anything up to
    an hour of standby card and promotion, and a sample taken there says nothing
    about the camera.
    """
    playlist = playlist_url(url, fmt)
    if not playlist:
        return []
    base, init, entries, total = segment_index(playlist)
    if not entries:
        return []

    begin = total * skip_fraction
    step = (total - begin) / max(1, samples)
    workdir = workdir or tempfile.mkdtemp(prefix="carom_screen_")
    os.makedirs(workdir, exist_ok=True)

    frames, used = [], set(exclude)
    for index in range(samples):
        wanted = begin + index * step
        start, name = min(entries, key=lambda entry: abs(entry[0] - wanted))
        if start in used:
            continue
        used.add(start)
        path = os.path.join(workdir, f"segment_{index}.m4s")
        try:
            with open(path, "wb") as handle:
                handle.write(init)
                handle.write(_get(name))
            capture = cv2.VideoCapture(path)
            ok, frame = capture.read()
            capture.release()
        except Exception:
            continue
        finally:
            if os.path.exists(path):
                os.remove(path)
        if ok:
            frames.append((start, frame))
    return frames


def judge(frames):
    """Turn sampled frames into a verdict."""
    calibrated, errors, scales, table_scales, with_table = 0, [], [], [], 0
    for _, frame in frames:
        quad = detect_cloth_quad(frame)
        if quad is not None and looks_like_a_table(quad):
            with_table += 1
            table_scales.append(table_scale(frame, quad))
        try:
            found = calibrate(frame)
        except CalibrationError:
            continue
        calibrated += 1
        errors.append(found.reprojection_error)
        scales.append(found.mm_per_px)

    seen = len(frames)
    verdict = {
        "usable": calibrated >= 2,
        "samples": seen,
        "calibrated": calibrated,
        "calibrated_share": calibrated / seen if seen else 0.0,
        "table_share": with_table / seen if seen else 0.0,
        "table_scale": sorted(table_scales)[len(table_scales) // 2] if table_scales else None,
        "median_error_mm": sorted(errors)[len(errors) // 2] if errors else None,
        "mm_per_px": sorted(scales)[len(scales) // 2] if scales else None,
    }
    if not verdict["usable"]:
        # Three outcomes, not two. A camera that never shows the whole table
        # cannot be fixed and the match is out; a match that simply has little
        # table time is only poor value; but a sample that shows the table over
        # and over without ever calibrating has told us nothing yet, and the
        # match deserves a closer look rather than a rejection.
        if calibrated:
            verdict["reason"] = "table rarely in frame"
        elif with_table >= max(2, seen // 4):
            verdict["reason"] = "undecided: table visible but not calibrated"
            verdict["undecided"] = True
        else:
            verdict["reason"] = "no overhead camera"
    return verdict


def assess(video_path, samples=DEFAULT_SAMPLES, skip_fraction=0.25):
    """Judge a video file already on disk. The fallback path, and how tests run."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    if duration <= 0:
        capture.release()
        return {"usable": False, "reason": "unreadable", "calibrated": 0, "samples": 0}

    begin = duration * skip_fraction
    step = (duration - begin) / max(1, samples)
    frames = []
    for index in range(samples):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int((begin + index * step) * fps))
        ok, frame = capture.read()
        if ok:
            frames.append((begin + index * step, frame))
    capture.release()
    return judge(frames)


def screen(url, workdir=None, keep_preview=False, verbose=True, samples=DEFAULT_SAMPLES):
    """Judge a match without downloading it.

    Segments are fetched at the broadcast's own resolution and thrown away one
    at a time, so screening costs tens of megabytes. Only if the playlist
    cannot be indexed does this fall back to downloading the small rendition
    whole - which is slower and, on a table shot from far back, can be wrong.
    """
    details = probe(url) or {}
    frames = sample_frames(url, samples=samples, workdir=workdir)
    if frames:
        verdict = judge(frames)
        # These matches spend most of their running time on standby cards,
        # replays and close-ups, so even thirty samples can leave only a
        # handful of frames of whole table, and the verdict turns on whether
        # one of them happened to calibrate. A single calibrated frame is
        # already proof the camera exists, and so is the table being in shot
        # over and over; either says the sample was too thin to decide, and the
        # answer to that is more samples rather than a rejection.
        if not verdict["usable"] and (verdict["calibrated"] or verdict["table_share"] >= 0.12):
            frames += sample_frames(url, samples=DEEP_SAMPLES, workdir=workdir,
                                    exclude={start for start, _ in frames})
            verdict = judge(frames)
            verdict["deep"] = True
    else:
        verdict = _screen_by_preview(url, workdir, keep_preview, details)
    verdict.update({"url": url, **details})
    if verbose:
        mark = "USE " if verdict["usable"] else "skip"
        extra = (f"calibrated {verdict['calibrated']}/{verdict['samples']}, "
                 f"{verdict['median_error_mm']:.1f} mm"
                 if verdict["usable"] else verdict.get("reason", ""))
        print(f"  {mark} {details.get('title', url)[:52]:52s} {extra}", flush=True)
    return verdict


def _screen_by_preview(url, workdir, keep_preview, details):
    """Download the small rendition whole and judge that. Fallback only."""
    workdir = workdir or tempfile.gettempdir()
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"screen_{details.get('id', 'x')}.mp4")
    try:
        if not fetch_preview(url, path):
            return {"usable": False, "reason": "preview download failed",
                    "calibrated": 0, "samples": 0}
        return assess(path)
    finally:
        if not keep_preview and os.path.exists(path):
            os.remove(path)
