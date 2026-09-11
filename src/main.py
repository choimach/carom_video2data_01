import os
import argparse
import json
from src.db.database import SessionLocal, init_db
from src.db.models import Match, Inning, Shot
from src.physics.ball_tracker import BallTracker
from src.physics.perspective import PerspectiveTransformer
from src.physics.kinematics import KinematicsEngine
from src.visualization.export_data import export_trajectories_to_json

def process_video(video_path: str, title: str):
    print(f"Starting v002 test pipeline for: {title}")
    print(f"Processing clip: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    # 1. Database Session
    db = SessionLocal()
    try:
        match = Match(
            title=title,
            video_url="local_test_clip",
            tournament_name="Test PBA Tour",
            player_a="Player A",
            player_b="Player B"
        )
        db.add(match)
        db.commit()
        db.refresh(match)
        
        inning = Inning(match_id=match.id, inning_number=1, player_id="Player A")
        db.add(inning)
        db.commit()
        db.refresh(inning)

        # 2. Vision & Physics Tracking
        print("Step 1: Running YOLO Ball Tracker...")
        tracker = BallTracker('yolov8n.pt')
        # This will create output_tracked.mp4 for visual verification
        trajectories = tracker.track_video(video_path, output_path="data/videos/output_tracked.mp4")
        
        # We need to map YOLO tracking IDs to 'white', 'yellow', 'red'.
        # Since yolov8n detects balls as 'sports ball', IDs are arbitrary integers.
        # For this test, we just pick the top 3 longest tracks.
        sorted_ids = sorted(trajectories.keys(), key=lambda k: len(trajectories[k]), reverse=True)
        if len(sorted_ids) < 3:
            print("Warning: Could not detect 3 distinct balls across the video.")
            
        white_pixels = trajectories[sorted_ids[0]] if len(sorted_ids) > 0 else []
        yellow_pixels = trajectories[sorted_ids[1]] if len(sorted_ids) > 1 else []
        red_pixels = trajectories[sorted_ids[2]] if len(sorted_ids) > 2 else []

        print("Step 2: Coordinate Transform (Perspective)")
        transformer = PerspectiveTransformer()
        # V002 Hardcoded Table Corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        # These assume a typical wide broadcast view. Will need adjustment per video.
        # Assuming video is 1920x1080 standard.
        transformer.calculate_matrix([(200, 200), (1720, 200), (1800, 900), (100, 900)])
        
        white_mm = transformer.transform_trajectory(white_pixels)
        yellow_mm = transformer.transform_trajectory(yellow_pixels)
        red_mm = transformer.transform_trajectory(red_pixels)

        print("Step 3: Kinematics Engine (Speed & Physics)")
        kinematics = KinematicsEngine(fps=30.0) # Using 30 fps for test clip
        speed = kinematics.calculate_speed(white_mm)
        print(f"Calculated Cue Ball Speed: {speed:.2f} m/s")

        # 3. Save to DB
        shot = Shot(
            inning_id=inning.id,
            shot_number=1,
            success=False, # We don't have scoring logic yet
            cue_speed=speed,
            thickness=0.0, # Placeholder
            spin_x=0.0,    # Placeholder
            spin_y=0.0     # Placeholder
        )
        db.add(shot)
        db.commit()
        db.refresh(shot)
        
        # Export logic to update visualization JSON
        export_data = {
            "table": {"width": 1422, "height": 2844},
            "shots": [
                {
                    "shot_id": shot.id,
                    "fps": 30,
                    "balls": {
                        "white": [{"x": p[0], "y": p[1]} for p in white_mm],
                        "yellow": [{"x": p[0], "y": p[1]} for p in yellow_mm],
                        "red": [{"x": p[0], "y": p[1]} for p in red_mm]
                    }
                }
            ]
        }
        
        json_path = "data/trajectories.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)
            
        print("Pipeline execution complete! Tracked video and JSON data exported.")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Billiards Video to Data Pipeline (v002)")
    parser.add_argument("--video", type=str, help="Path to local video clip", default="data/videos/test_shot.mp4")
    parser.add_argument("--title", type=str, help="Match Title", default="Test Clip PBA")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database")
    
    args = parser.parse_args()
    
    if args.init_db:
        init_db()
        print("Database initialized.")
        
    process_video(args.video, args.title)
