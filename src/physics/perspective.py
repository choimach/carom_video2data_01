import cv2
import numpy as np

class PerspectiveTransformer:
    """Pixel -> millimetre mapping for a single camera setup.

    Note on orientation: this class defaults to a portrait table (x across the
    1422 mm width, y along the 2844 mm length), which is what
    visualization/export_data.py and viewer.html expect. A Calibration produced
    by table_calibration is landscape (x along the length), matching how the
    overhead camera frames the table; from_calibration() therefore yields a
    transformer whose output is landscape, and anything downstream that assumes
    portrait has to swap the axes.
    """

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

    @classmethod
    def from_calibration(cls, calibration):
        """Build a transformer from a diamond-based Calibration.

        Prefer this over calculate_matrix(): hand-picked table corners are read
        off the cloth edge, which sits about 83 mm outside the cushion nose line
        on each rail and biases the scale by roughly 3%.
        """
        from src.physics.table_calibration import TABLE_LENGTH_MM, TABLE_WIDTH_MM

        transformer = cls(table_width_mm=TABLE_LENGTH_MM, table_height_mm=TABLE_WIDTH_MM)
        transformer.matrix = calibration.matrix
        transformer.calibration = calibration
        return transformer

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
