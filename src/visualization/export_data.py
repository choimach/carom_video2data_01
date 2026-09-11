import os
import json
from src.db.database import SessionLocal
from src.db.models import Shot

def export_trajectories_to_json(output_path="data/trajectories.json"):
    """
    Exports saved shot trajectories from the database to a JSON file
    for the web-based visualization viewer.
    """
    # For demonstration, we will generate a dummy trajectory that follows 
    # a typical 3-cushion path, since our DB currently holds a dummy Shot record.
    
    # Coordinates are in mm. Table is 1422 x 2844.
    dummy_data = {
        "table": {
            "width": 1422,
            "height": 2844
        },
        "shots": [
            {
                "shot_id": 1,
                "fps": 60,
                "balls": {
                    "white": [
                        {"x": 711, "y": 711}, # Start
                        {"x": 1000, "y": 1400},
                        {"x": 1422, "y": 2000}, # Hit right cushion
                        {"x": 711, "y": 2844},  # Hit bottom cushion
                        {"x": 0, "y": 2000},    # Hit left cushion
                        {"x": 711, "y": 2133}   # Final position
                    ],
                    "yellow": [
                        {"x": 711, "y": 2133},
                        {"x": 700, "y": 2200}
                    ],
                    "red": [
                        {"x": 300, "y": 2000},
                        {"x": 350, "y": 2050}
                    ]
                }
            }
        ]
    }
    
    # In a real scenario, query the DB:
    # db = SessionLocal()
    # shots = db.query(Shot).all()
    # # ... process shots into JSON
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, indent=4)
        
    print(f"Exported data to {output_path}")

if __name__ == "__main__":
    export_trajectories_to_json()
