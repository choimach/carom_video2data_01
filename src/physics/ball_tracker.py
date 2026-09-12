import cv2
import numpy as np
from ultralytics import YOLO

class BallTracker:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLO tracker for billiard balls.
        For production, a fine-tuned YOLO model specifically trained on 
        billiard balls (white, yellow, red) should be loaded here instead 
        of the default yolov8n.pt.
        """
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        
        # Color definitions for drawing bounding boxes (BGR format)
        self.colors = {
            'white': (255, 255, 255),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'unknown': (0, 255, 0)
        }

    def detect_balls(self, frame):
        """
        Detect balls in a single frame.
        Returns a list of dictionaries with bounding boxes and classes.
        """
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            # Get coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # NOTE: In a custom-trained model, cls_id would map to 'white', 'yellow', 'red'
            # For now, we just record the raw data
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls_id
            })
            
        return detections

    def track_video(self, video_path, output_path=None, max_frames=None):
        """
        Track balls across a video and optionally save the annotated video.
        Uses YOLO's built-in tracker (BoT-SORT / ByteTrack).
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        trajectories = {}
        frame_count = 0

        while cap.isOpened():
            if max_frames and frame_count >= max_frames:
                break
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
                
            # Run YOLO tracking (persist tracking IDs across frames)
            results = self.model.track(frame, persist=True, verbose=False)[0]
            
            annotated_frame = frame.copy()
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    trajectories[track_id].append((cx, cy))
                    
                    # Draw box and ID
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.colors['unknown'], 2)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['unknown'], 2)
            
            if output_path:
                out.write(annotated_frame)
                
        cap.release()
        if output_path:
            out.release()
            
        return trajectories

if __name__ == "__main__":
    # Example usage:
    # tracker = BallTracker()
    # traj = tracker.track_video("sample_shot.mp4", "output_tracked.mp4")
    # print(f"Tracked {len(traj)} objects.")
    pass
