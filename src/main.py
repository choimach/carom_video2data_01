import os
import argparse
from src.db.database import SessionLocal, init_db
from src.db.models import Match, Inning, Shot
from src.data_acquisition.downloader import download_soop_video
# from src.segmentation.scoreboard_ocr import ScoreboardReader
# from src.physics.ball_tracker import BallTracker
# from src.physics.perspective import PerspectiveTransformer
# from src.physics.kinematics import KinematicsEngine

def process_video(url: str, title: str):
    print(f"Starting pipeline for: {title}")
    
    # 1. Database Session
    db = SessionLocal()
    try:
        # Create a new Match entry
        match = Match(
            title=title,
            video_url=url,
            tournament_name="Unknown Tournament",
            player_a="Player A",
            player_b="Player B"
        )
        db.add(match)
        db.commit()
        db.refresh(match)
        print(f"Created Match Record ID: {match.id}")
        
        # 2. Data Acquisition
        print("Step 1: Downloading video...")
        output_dir = "data/videos"
        # Download function will save the video. For a full pipeline, we'd capture the exact file path.
        # success = download_soop_video(url, output_dir=output_dir)
        
        print("Step 2: Video Segmentation (OCR & Shot Split) - [Stub]")
        # reader = ScoreboardReader()
        # ... Run OCR on frames, split into innings and shots
        
        print("Step 3: Vision & Physics Tracking - [Stub]")
        # tracker = BallTracker()
        # transformer = PerspectiveTransformer()
        # kinematics = KinematicsEngine()
        # ... For each shot clip, track balls, transform, and calculate physics
        
        # 3. Save Dummy Result to DB (To demonstrate pipeline flow)
        inning = Inning(match_id=match.id, inning_number=1, player_id="Player A")
        db.add(inning)
        db.commit()
        db.refresh(inning)
        
        dummy_shot = Shot(
            inning_id=inning.id,
            shot_number=1,
            success=True,
            cue_speed=2.5,
            thickness=50.0,
            spin_x=1.2,
            spin_y=-0.5
        )
        db.add(dummy_shot)
        db.commit()
        
        print("Pipeline execution complete! Data saved to SQLite database.")
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Billiards Video to Data Pipeline")
    parser.add_argument("--url", type=str, help="SOOP VOD URL", required=True)
    parser.add_argument("--title", type=str, help="Match Title", default="Sample Match")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database")
    
    args = parser.parse_args()
    
    if args.init_db:
        init_db()
        print("Database initialized.")
        
    process_video(args.url, args.title)
