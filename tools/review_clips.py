"""Cut a sample of plays for a human to judge, so a new event can be checked.

The 94% the labelling reports is measured on one match - the only one with
labels written by hand. Whether it holds on an event the pipeline has never
seen is a different question, and the only way to answer it is to look.

Two samples, for two different questions:

* the plays where the inning's shape and the cue ball's path disagree. This is
  where the errors live, and judging these says which witness is wrong when.
* a plain random sample of usable plays, which is the only one that gives an
  honest accuracy figure.

Clips are named by number and time only. What the pipeline decided is in
index.csv and not on the filename, because a reviewer who can see the answer is
no longer an independent check.

    ~/.venvs/carom/bin/python tools/review_clips.py POWC2026 --count 12
"""

import argparse
import csv
import glob
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import analyse, load_scan  # noqa: E402


def cut(video, begin, finish, path, lead_in=4.0, lead_out=7.0, width=960, crf=28):
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, begin - lead_in):.2f}",
        "-i", video, "-t", f"{(finish - begin) + lead_in + lead_out:.2f}",
        "-an", "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", str(crf),
        "-preset", "veryfast", path, "-y",
    ]
    return subprocess.run(command, capture_output=True, text=True).returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cut plays for human review")
    parser.add_argument("event", help="part of the scan filename, e.g. POWC2026")
    parser.add_argument("--count", type=int, default=12, help="clips per sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    scans = [p for p in sorted(glob.glob(os.path.join(ROOT, "data", "scans", "*.npz")))
             if args.event in p]
    if not scans:
        print(f"no scan matching {args.event}", file=sys.stderr)
        return 1
    scan_path = scans[0]
    name = os.path.splitext(os.path.basename(scan_path))[0]
    video = os.path.join(ROOT, "data", "videos", f"{name}.mp4")
    directory = args.out or os.path.join(ROOT, "data", "review", name)
    os.makedirs(directory, exist_ok=True)

    result = analyse(load_scan(scan_path))
    start, fps = result["start"], result["fps"]
    usable = [(inn, n, s) for inn in result["innings"]
              for n, s in enumerate(inn.shots, start=1)
              if not s.inferred and not getattr(s, "rejections", None)]

    disputed = [r for r in usable
                if r[2].inning_success is not None
                and r[2].trajectory_success is not None
                and r[2].inning_success != r[2].trajectory_success]
    random.Random(args.seed).shuffle(disputed)
    rest = [r for r in usable if r not in disputed]
    random.Random(args.seed).shuffle(rest)

    chosen = ([("disputed", r) for r in disputed[:args.count]]
              + [("random", r) for r in rest[:args.count]])
    chosen.sort(key=lambda pair: pair[1][2].start_frame)

    rows = []
    for index, (sample, (inning, number, shot)) in enumerate(chosen, start=1):
        begin = start + shot.start_frame / fps
        finish = start + shot.end_frame / fps
        clip = f"p{index:03d}_t{int(begin)}.mp4"
        if not cut(video, begin, finish, os.path.join(directory, clip)):
            print(f"  clip failed: {clip}", file=sys.stderr)
            continue
        rows.append({
            "clip": clip, "sample": sample, "t_start": round(begin, 1),
            "inning": inning.number, "shot": number, "cue": shot.cue_ball,
            "pipeline": "SCORE" if shot.success else "MISS",
            "from": shot.verdict_source or "",
            "inning_says": shot.inning_success,
            "trajectory_says": shot.trajectory_success,
            "cushions": (shot.verdict or {}).get("cushions"),
            "thickness": None if shot.thickness is None else round(shot.thickness, 2),
        })
        print(f"  {clip}  {sample}", flush=True)

    with open(os.path.join(directory, "index.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    template = os.path.join(directory, "feedback.txt")
    if not os.path.exists(template):
        with open(template, "w", encoding="utf-8") as handle:
            handle.write("각 클립을 보고 성공/실패만 적어주세요. 파이프라인 판정은 "
                         "index.csv 에 있고 파일명에는 없습니다.\n\n")
            for row in rows:
                handle.write(f"({row['clip'][:-4]})\n\n")
    print(f"\n{len(rows)} clips in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
