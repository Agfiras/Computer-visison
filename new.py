import cv2
import numpy as np
import pyautogui
import time
from scipy.spatial import distance as dist

# Constants for skin detection
SKIN_COLOR_LOWER = np.array([0, 48, 80], dtype=np.uint8)
SKIN_COLOR_UPPER = np.array([20, 255, 255], dtype=np.uint8)

# Homography target rectangle size
RECT_SIZE = (300, 200)
CURSOR_SMOOTHING_FACTOR = 0.3  # To reduce jitter

# Get screen dimensions for full-screen calibration
screen_width, screen_height = pyautogui.size()

# Predefined target points on the screen for calibration
target_points = [
    (100, 100), 
    (screen_width - 100, 100), 
    (screen_width - 100, screen_height - 100), 
    (100, screen_height - 100)
]

def detect_edges(frame, method="original"):
    """
    Detect edges using Canny.
    Supports both 'original' and 'optimized' methods.
    :param frame: Input frame for edge detection.
    :param method: 'original' for plain Canny, 'optimized' for enhanced quality.
    :return: Edges detected in the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if method == "original":
        # Original Canny Edge Detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
    
    elif method == "optimized":
        # Optimized Canny Edge Detection
        # Step 1: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Step 2: Noise reduction using bilateral filter
        gray_filtered = cv2.bilateralFilter(gray_enhanced, 9, 75, 75)
        
        # Step 3: Dynamic thresholding for Canny
        median_intensity = np.median(gray_filtered)
        lower_threshold = int(max(0, 0.66 * median_intensity))
        upper_threshold = int(min(255, 1.33 * median_intensity))
        edges = cv2.Canny(gray_filtered, lower_threshold, upper_threshold)
    
    else:
        raise ValueError("Invalid method. Choose 'original' or 'optimized'.")
    
    return edges


def detect_touchpad(edges):
    """Detect touchpad by finding quadrilaterals in contours."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 1000:
            return approx
    return None


def compute_homography(touchpad):
    """Compute homography matrix for touchpad."""
    rect = np.array([[0, 0], [RECT_SIZE[0], 0], [RECT_SIZE[0], RECT_SIZE[1]], [0, RECT_SIZE[1]]], dtype="float32")
    return cv2.findHomography(touchpad, rect)[0]

    print("Homography Matrix:\n", homography_matrix)
    
    return homography_matrix


def segment_skin(frame):
    """Segment skin using HSV color space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_COLOR_LOWER, SKIN_COLOR_UPPER)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def detect_fingertip(contour):
    """Detect the fingertip from the largest contour."""
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return None

    farthest_point = None
    max_distance = -1
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        farthest = tuple(contour[f][0])
        if d > max_distance and d > 5000:  # Minimum defect depth to filter noise
            max_distance = d
            farthest_point = farthest

    return farthest_point


def smooth_cursor(current, previous):
    """Smooth cursor movement to reduce jitter."""
    if previous is None:
        return current
    return (
        int(CURSOR_SMOOTHING_FACTOR * current[0] + (1 - CURSOR_SMOOTHING_FACTOR) * previous[0]),
        int(CURSOR_SMOOTHING_FACTOR * current[1] + (1 - CURSOR_SMOOTHING_FACTOR) * previous[1])
    )


def main():
    cap = cv2.VideoCapture(0)
    touchpad = None
    homography_matrix = None
    previous_cursor_position = None
    last_cursor_position = (screen_width // 2, screen_height // 2)  # Initial cursor position (center)

    target_index = 0  # Index for current target point

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if touchpad is None:
            # Use either 'original' or 'optimized' for edge detection
            edges = detect_edges(frame, method="optimized")  # Change to "original" for comparison
            touchpad = detect_touchpad(edges)
            if touchpad is not None:
                homography_matrix = compute_homography(touchpad)
                print("Touchpad detected and homography computed.")
        else:
            transformed_frame = cv2.warpPerspective(frame, homography_matrix, RECT_SIZE)
            skin_mask = segment_skin(transformed_frame)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 2000:
                    fingertip = detect_fingertip(largest_contour)
                    if fingertip:
                        # Draw fingertip for visualization
                        cv2.circle(transformed_frame, fingertip, 5, (255, 0, 0), -1)

                        # Calculate cursor position ratios
                        x_ratio = fingertip[0] / RECT_SIZE[0]
                        y_ratio = fingertip[1] / RECT_SIZE[1]

                        # Map touchpad coordinates to the full screen size
                        cursor_position = (x_ratio * screen_width, y_ratio * screen_height)

                        # Smooth cursor movement
                        smoothed_cursor_position = smooth_cursor(cursor_position, previous_cursor_position)

                        # If finger moves out of touchpad, keep last position until reintroduced
                        if dist.euclidean(smoothed_cursor_position, last_cursor_position) > 0:
                            last_cursor_position = smoothed_cursor_position

                        pyautogui.moveTo(*last_cursor_position)
                        previous_cursor_position = smoothed_cursor_position
                        print(f"Cursor moved to: {int(last_cursor_position[0])}, {int(last_cursor_position[1])}")

            # Calibration: Move cursor to predefined target points
            if dist.euclidean(last_cursor_position, target_points[target_index]) < 10:
                target_index = (target_index + 1) % len(target_points)  # Move to next target point
                pyautogui.moveTo(target_points[target_index])
                print(f"Moved to target: {target_points[target_index]}")

            # Draw detected touchpad
            cv2.drawContours(frame, [touchpad], -1, (0, 255, 0), 2)
        cv2.imshow("Canny edge", detect_edges(frame))
        cv2.imshow("Virtual Touchpad", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")


main() 