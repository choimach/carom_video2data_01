"""
Turn a match video into play data.

    python -m src.main --video data/videos/match.mp4 --title "Jaspers vs Tran"
    python -m src.main --url https://vod.sooplive.com/player/206343559
    python -m src.main --video ... --analyse-only      # re-run on the cached scan

The scan is the expensive half, so it is written to a .npz beside the video and
reused. Every threshold in the judgement was settled by re-running --analyse-only
over a cached scan; at fifty minutes a scan there is no other way to work.
"""

import argparse
import os
import sys

from src.db.database import SessionLocal, init_db, reset_db
from src.db.models import Inning, Match, Shot
from src.pipeline import (analyse, cut_clips, export_json, export_trajectories,
                          find_calibration, find_play_start, load_scan, report, scan)


def track_path_for(video_path):
    base, _ = os.path.splitext(video_path)
    return f"{base}.track.npz"


def store(result, video_path, track_path, title, url, scan_data):
    """Write the match, its innings and its plays to the database."""
    session = SessionLocal()
    try:
        turns = result["turns"]
        colours = {colour for *_rest, colour, _points in
                   ((t[0], t[1], t[2], t[3]) for t in turns)}
        match = Match(
            title=title,
            video_url=url,
            video_path=video_path,
            track_path=track_path,
            track_start_second=result["start"],
            fps=result["fps"],
            mm_per_px=scan_data["mm_per_px"],
            calibration_error_mm=scan_data["reprojection_error"],
            player_a_ball="white" if "white" in colours else None,
            player_b_ball="yellow" if "yellow" in colours else None,
            expected_plays=result["expected_plays"],
            detected_plays=len(result["plays"]),
        )
        session.add(match)
        session.flush()

        for inning, turn in zip(result["innings"], turns):
            points = turn[3]
            row = Inning(
                match_id=match.id,
                inning_number=inning.number,
                cue_ball=inning.cue_ball,
                player_id=inning.cue_ball,
                points=points,
                expected_plays=points + 1,
                complete=len(inning.shots) == points + 1,
            )
            session.add(row)
            session.flush()
            for number, shot in enumerate(inning.shots, start=1):
                verdict = getattr(shot, "verdict", {}) or {}
                layout = shot.start_positions or {}
                rejections = getattr(shot, "rejections", [])
                session.add(Shot(
                    inning_id=row.id,
                    shot_number=number,
                    cue_ball=shot.cue_ball,
                    success=bool(shot.success),
                    verdict_source="trajectory",
                    scoreboard_success=getattr(shot, "scoreboard_success", None),
                    cushions_before_second=verdict.get("cushions"),
                    first_object_ball=verdict.get("first_ball"),
                    second_object_ball=verdict.get("second_ball"),
                    white_x=layout.get("white", (None, None))[0],
                    white_y=layout.get("white", (None, None))[1],
                    yellow_x=layout.get("yellow", (None, None))[0],
                    yellow_y=layout.get("yellow", (None, None))[1],
                    red_x=layout.get("red", (None, None))[0],
                    red_y=layout.get("red", (None, None))[1],
                    end_white_x=(shot.end_positions or {}).get("white", (None, None))[0],
                    end_white_y=(shot.end_positions or {}).get("white", (None, None))[1],
                    end_yellow_x=(shot.end_positions or {}).get("yellow", (None, None))[0],
                    end_yellow_y=(shot.end_positions or {}).get("yellow", (None, None))[1],
                    end_red_x=(shot.end_positions or {}).get("red", (None, None))[0],
                    end_red_y=(shot.end_positions or {}).get("red", (None, None))[1],
                    video_clip_path=getattr(shot, "clip_path", None),
                    cue_speed=getattr(shot, "cue_speed", None),
                    cue_travel_mm=getattr(shot, "cue_travel", None),
                    start_frame=shot.start_frame,
                    end_frame=shot.end_frame,
                    start_second=result["start"] + shot.start_frame / result["fps"],
                    end_second=result["start"] + shot.end_frame / result["fps"],
                    complete=shot.complete,
                    inferred=shot.inferred,
                    is_replay=shot.replay_of is not None,
                    label_confirmed=getattr(shot, "label_confirmed", False),
                    usable=not rejections,
                    rejected_for=";".join(rejections) or None,
                ))
        session.commit()
        return match.id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract carom plays from a match video")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="path to a match video")
    source.add_argument("--url", help="match VOD to download first")
    parser.add_argument("--title", default=None, help="name for this match")
    parser.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--no-seek", action="store_true",
                        help="scan from --start rather than hunting for where the match begins")
    parser.add_argument("--analyse-only", action="store_true",
                        help="reuse the cached scan instead of reading the video again")
    parser.add_argument("--no-store", action="store_true", help="do not write to the database")
    parser.add_argument("--json", default=None, help="also write the plays to this JSON file")
    parser.add_argument("--trajectories", default=None,
                        help="write every usable play's three ball paths to this .npz")
    parser.add_argument("--clips", default=None,
                        help="cut one video file per usable play into this directory")
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--reset-db", action="store_true", help="drop and recreate the tables")
    args = parser.parse_args(argv)

    if args.reset_db:
        reset_db()
        print("database reset")
    elif args.init_db:
        init_db()
        print("database initialised")
    if not (args.video or args.url):
        if args.init_db or args.reset_db:
            return 0
        parser.error("one of --video or --url is required")

    video_path = args.video
    if args.url:
        from src.data_acquisition.downloader import download_soop_video

        if not download_soop_video(args.url):
            print("download failed", file=sys.stderr)
            return 1
        if not video_path:
            parser.error("--url downloaded the video; re-run with --video pointing at it")
    if not os.path.exists(video_path):
        print(f"no such video: {video_path}", file=sys.stderr)
        return 1

    init_db()
    track_path = track_path_for(video_path)
    if not args.analyse_only or not os.path.exists(track_path):
        if args.analyse_only:
            print(f"no cached scan at {track_path}; scanning", flush=True)
        calibration, _fps = find_calibration(video_path, search_from=max(args.start, 60.0))
        start = args.start
        if not args.no_seek:
            found = find_play_start(video_path, calibration, search_from=args.start)
            if found is not None:
                start = max(args.start, found - 30.0)  # a little room before the first shot
        scan(video_path, track_path, start=start, end=args.end, calibration=calibration)
    else:
        print(f"using cached scan {track_path}", flush=True)

    scan_data = load_scan(track_path)
    result = analyse(scan_data)
    title = args.title or os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{title}")
    report(result)

    if args.clips:
        clips = cut_clips(video_path, result, args.clips)
        print(f"\ncut {len(clips)} clips into {args.clips}")
    if args.trajectories:
        count = export_trajectories(scan_data, result, args.trajectories)
        print(f"wrote {count} plays' trajectories to {args.trajectories}")
    if args.json:
        count = export_json(result, args.json)
        print(f"wrote {count} plays to {args.json}")
    if not args.no_store:
        match_id = store(result, video_path, track_path, title, args.url, scan_data)
        print(f"stored as match {match_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
