import cv2
import easyocr
import numpy as np

class ScoreboardReader:
    def __init__(self, lang_list=['en']):
        # Initialize EasyOCR reader
        # Using English by default as scores/innings are usually numbers
        self.reader = easyocr.Reader(lang_list)
        
    def preprocess_image(self, frame, roi=None):
        """
        Crop the Region of Interest (ROI), convert to grayscale, and threshold 
        to improve OCR accuracy for digital scoreboard numbers.
        """
        if roi:
            x, y, w, h = roi
            img = frame[y:y+h, x:x+w]
        else:
            img = frame
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle different lighting on scoreboards
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        return thresh

    def read_text(self, frame, roi=None):
        """
        Read text from a specific ROI in the frame.
        Returns a list of detected strings.
        """
        processed_img = self.preprocess_image(frame, roi)
        
        # reader.readtext returns a list of tuples: (bbox, text, confidence)
        results = self.reader.readtext(processed_img)
        
        detected_texts = []
        for (bbox, text, prob) in results:
            if prob > 0.5: # Confidence threshold
                detected_texts.append(text)
                
        return detected_texts

    def detect_inning(self, frame, inning_roi):
        """
        Specific function to parse the current inning number.
        """
        texts = self.read_text(frame, inning_roi)
        # Often looks like "Inning 4" or just "4"
        for t in texts:
            # simple digit extraction
            digits = ''.join(filter(str.isdigit, t))
            if digits:
                return int(digits)
        return None

if __name__ == "__main__":
    # Example usage:
    # reader = ScoreboardReader()
    # cap = cv2.VideoCapture("sample_video.mp4")
    # ret, frame = cap.read()
    # # Example ROI: top-left corner (x, y, w, h)
    # roi = (10, 10, 200, 100) 
    # print(reader.read_text(frame, roi))
    pass
