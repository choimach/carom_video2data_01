import numpy as np
import math

class KinematicsEngine:
    def __init__(self, fps=60.0):
        self.fps = fps
        # Ball diameter for carom (approx 61.5mm)
        self.ball_diameter_mm = 61.5
        
    def calculate_speed(self, trajectory_mm):
        """
        Calculates the initial cue speed based on the first few frames 
        of the cue ball trajectory.
        trajectory_mm: list of (x, y) tuples in millimeter scale.
        """
        if len(trajectory_mm) < 2:
            return 0.0
            
        # Use the first 5 frames to calculate initial speed (to avoid deceleration)
        points = trajectory_mm[:min(5, len(trajectory_mm))]
        
        total_distance = 0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_distance += math.hypot(dx, dy)
            
        # Distance per frame * frames per second = Distance per second (mm/s)
        # Convert to m/s
        avg_dist_per_frame = total_distance / (len(points) - 1)
        speed_m_s = (avg_dist_per_frame * self.fps) / 1000.0
        
        return speed_m_s

    def calculate_thickness(self, cb_pre_hit, ob_post_hit):
        """
        Estimates the hit thickness based on the Object Ball's (OB) exit angle.
        This relies on the "90-degree rule" / "30-degree rule" mechanics.
        
        cb_pre_hit: Vector (dx, dy) of the cue ball before impact.
        ob_post_hit: Vector (dx, dy) of the object ball after impact.
        """
        # Normalize vectors
        norm_cb = np.linalg.norm(cb_pre_hit)
        norm_ob = np.linalg.norm(ob_post_hit)
        
        if norm_cb == 0 or norm_ob == 0:
            return None
            
        cb_vec = np.array(cb_pre_hit) / norm_cb
        ob_vec = np.array(ob_post_hit) / norm_ob
        
        # Calculate the angle between cue ball trajectory and object ball trajectory
        cos_theta = np.clip(np.dot(cb_vec, ob_vec), -1.0, 1.0)
        angle_rad = math.acos(cos_theta)
        
        # In an elastic collision, if OB goes straight forward (angle=0), it's 100% full hit.
        # If OB goes at 90 degrees (pi/2), it's a 0% hit (miss).
        # We can map this angle to thickness.
        # sin(angle) relates to the offset distance.
        offset_ratio = math.sin(angle_rad)
        thickness_percentage = (1.0 - offset_ratio) * 100.0
        
        # Cap to 0-100%
        return max(0.0, min(100.0, thickness_percentage))
        
    def estimate_spin(self, pre_cushion_vec, post_cushion_vec, normal_vec):
        """
        Estimates side spin (English) based on how the reflection angle 
        deviates from the law of reflection (angle of incidence = angle of reflection).
        """
        # This is a simplified estimation based on angle deviations
        # A more rigorous implementation would use Han (2005) math formulas.
        # Normal vector points out from the cushion.
        
        # Calculate incidence angle
        norm_pre = np.linalg.norm(pre_cushion_vec)
        norm_post = np.linalg.norm(post_cushion_vec)
        
        if norm_pre == 0 or norm_post == 0:
            return 0.0
            
        pre_dir = np.array(pre_cushion_vec) / norm_pre
        post_dir = np.array(post_cushion_vec) / norm_post
        normal = np.array(normal_vec)
        
        incidence_cos = np.dot(-pre_dir, normal)
        reflection_cos = np.dot(post_dir, normal)
        
        angle_in = math.acos(np.clip(incidence_cos, -1.0, 1.0))
        angle_out = math.acos(np.clip(reflection_cos, -1.0, 1.0))
        
        # Difference in angles represents the English (spin) effect
        # Positive = Running English, Negative = Reverse English
        spin_deviation_rad = angle_out - angle_in
        spin_deviation_deg = math.degrees(spin_deviation_rad)
        
        # Map degrees to rough "Tips" (e.g. 1 tip ~ 5 degrees deviation)
        spin_tips = spin_deviation_deg / 5.0
        
        return spin_tips

if __name__ == "__main__":
    # Test
    ke = KinematicsEngine(fps=60.0)
    # 5 frames moving 50mm each frame -> 3000mm/s -> 3.0 m/s
    traj = [(0,0), (50,0), (100,0), (150,0), (200,0)]
    speed = ke.calculate_speed(traj)
    print(f"Speed: {speed} m/s")
