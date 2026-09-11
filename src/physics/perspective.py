import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, table_width_mm=1422.0, table_height_mm=2844.0):
        """
        Initializes the perspective transformer based on official carom billiard dimensions.
        Default: 1,422 mm x 2,844 mm (Full-size Match table)
        """
        self.width_mm = table_width_mm
        self.height_mm = table_height_mm
        
        # 2D table corner coordinates (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        # Using millimeter scale directly is very useful for physics calculations (m/s).
        self.dst_points = np.array([
            [0, 0],
            [self.width_mm, 0],
            [self.width_mm, self.height_mm],
            [0, self.height_mm]
        ], dtype=np.float32)
        
        self.matrix = None

    def calculate_matrix(self, src_points):
        """
        Computes the Homography matrix.
        src_points: List or array of 4 (x,y) tuples representing the 4 corners of the 
                    billiard table cushion edges found in the image frame.
                    Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        """
        src = np.array(src_points, dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(src, self.dst_points)
        return self.matrix

    def transform_point(self, x, y):
        """
        Transforms a single (x, y) point from camera pixel coordinates 
        to actual 2D millimeter table coordinates.
        """
        if self.matrix is None:
            raise ValueError("Homography matrix not calculated. Call calculate_matrix first.")
            
        pts = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.matrix)
        
        return dst[0][0][0], dst[0][0][1]

    def transform_trajectory(self, trajectory):
        """
        Transforms a list of (x, y) tuples.
        """
        return [self.transform_point(x, y) for x, y in trajectory]

if __name__ == "__main__":
    # Example test
    # pt = PerspectiveTransformer()
    # image_corners = [(100, 100), (900, 150), (950, 700), (50, 650)]
    # pt.calculate_matrix(image_corners)
    # print(pt.transform_point(500, 400))
    pass
