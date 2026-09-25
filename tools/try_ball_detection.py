"""고친 공 검출기를 **영상에서 바로** 재 본다 — 스캔을 다시 돌리기 전에.

경기 하나를 통째로 스캔하면 30분이 넘는다. 여기서는 스캔이 "테이블이 보인다"고
표시해 둔 프레임에서 몇백 장만 뽑아 검출률을 잰다. 1분이면 된다.

⚠️ 고친 쪽만 재지 말 것 (CLAUDE.md §7). 망가졌던 경기와 **멀쩡하던 경기**를
   같이 넣어야 한다.

    ~/.venvs/carom/bin/python tools/try_ball_detection.py <경기이름> [프레임수]
    ~/.venvs/carom/bin/python tools/try_ball_detection.py --all
"""
import os, sys, glob
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.physics.ball_detector import BallDetector          # noqa: E402
from src.physics.table_calibration import Calibration       # noqa: E402

BALLS = ("white", "yellow", "red")
LABEL = {"white": "흰", "yellow": "노", "red": "빨"}


def run(name, frames=240):
    scan_path = os.path.join(ROOT, "data", "scans", f"{name}.npz")
    video = os.path.join(ROOT, "data", "videos", f"{name}.mp4")
    if not (os.path.exists(scan_path) and os.path.exists(video)):
        print(f"  {name[:34]:<34}  스캔 또는 영상 없음")
        return
    scan = np.load(scan_path, allow_pickle=True)
    track, start, fps = scan["track"], float(scan["start"]), float(scan["fps"])
    calib = Calibration(
        matrix=scan["matrix"], corners=scan["corners"].astype(np.float32),
        mm_per_px=float(scan["mm_per_px"]), rail_offset_mm=(0.0, 0.0),
        reprojection_error=float(scan["reprojection_error"]), diamonds={},
    )
    live = np.flatnonzero(track[:, 0] == 1.0)
    if len(live) == 0:
        print(f"  {name[:34]:<34}  테이블이 보이는 프레임 없음")
        return
    # 경기 전체에 고루 흩어지게 (팔레트는 앞쪽에서 배우므로 순서대로 먹인다)
    picks = live[np.linspace(0, len(live) - 1, min(frames, len(live))).astype(int)]
    cap = cv2.VideoCapture(video)
    detector = BallDetector(calib)
    got = {c: 0 for c in BALLS}
    seen = three = 0
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps) + int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        seen += 1
        found = detector.detect(frame)
        for c in BALLS:
            if c in found:
                got[c] += 1
        if len(found) == 3:
            three += 1
    cap.release()
    if seen == 0:
        print(f"  {name[:34]:<34}  프레임을 못 읽음")
        return
    bar = "".join(f"{got[c] / seen:>6.0%}" for c in BALLS)
    learned = "팔레트O" if detector._palette else "팔레트X"
    print(f"  {name[:34]:<34}{bar}{three / seen:>7.0%}   {learned}  ({seen}장)")


def main(argv):
    names = argv
    if not argv or argv[0] == "--all":
        names = [os.path.basename(p)[:-4]
                 for p in sorted(glob.glob(os.path.join(ROOT, "data", "videos", "*.mp4")))]
    print(f"  {'경기':<34}" + "".join(f"{LABEL[c]:>6}" for c in BALLS) + f"{'셋다':>7}")
    for name in names:
        run(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
