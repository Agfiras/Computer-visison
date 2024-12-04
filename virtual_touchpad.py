import cv2
import numpy as np
import pyautogui

def edge_detection(frame): # Canny edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur
    edges = cv2.Canny(blurred, 50, 150) # Apply Canny edge detection
    return edges # Return the edges

def detect_touchpad(edges): # Detect the touchpad in the frame
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours
    for contour in contours: # Iterate through the contours
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True) # Approximate the contour
        if len(approx) == 4 and cv2.isContourConvex(approx): # Check if the contour is a quadrilateral
            area = cv2.contourArea(approx) # Compute the area of the contour
            if area > 1000:  # Filter out small contours
                return approx # Return the contour
    return None # Return None if no touchpad is found

def compute_homography(touchpad): # Compute the homography matrix
    rect = np.array([[0, 0], [300, 0], [300, 200], [0, 200]], dtype="float32")
    h, _ = cv2.findHomography(touchpad, rect)
    return h

def segment_skin(frame):# Segment the skin in the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV color space
    lower_skin = np.array([0, 43, 18], dtype=np.uint8) # Define lower
    upper_skin = np.array([20, 255, 255], dtype=np.uint8) # and upper
    mask = cv2.inRange(hsv, lower_skin, upper_skin) # Create a mask
    return mask # Return the mask

def detect_finger(mask): # Detect the finger in the segmented skin
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours
    if not contours: # Return None if no contours are found
        return None    

    largest_contour = max(contours, key=cv2.contourArea) # Find the largest contour
    hull = cv2.convexHull(largest_contour, returnPoints=False) # Compute the convex hull
    if len(hull) < 3: # Return None if the convex hull has less than 3 points
        return None

    try: # Compute the convexity defects
        defects = cv2.convexityDefects(largest_contour, hull) 
    except cv2.error as e: # Handle errors
        print(f"Error in convexityDefects: {e}")
        return None

    if defects is None: # Return None if no defects are found
        return None

    # Find the farthest point from the convex hull
    max_defect = max(defects, key=lambda x: x[0][3])
    s, e, f, d = max_defect[0]
    farthest_point = tuple(largest_contour[f][0])
    return farthest_point

def main():
    cap = cv2.VideoCapture(0)  # Start webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    screen_width, screen_height = pyautogui.size()
    touchpad = None
    homography_matrix = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            if touchpad is None:
                edges = edge_detection(frame)
                touchpad = detect_touchpad(edges)
                if touchpad is not None:
                    homography_matrix = compute_homography(touchpad)
                    print("Touchpad detected and homography computed.")
            else:
                transformed_frame = cv2.warpPerspective(frame, homography_matrix, (300, 200))
                skin = segment_skin(transformed_frame)
                finger_pos = detect_finger(skin)

                if finger_pos is not None:
                    cv2.circle(transformed_frame, finger_pos, 5, (255, 0, 0), -1)
                    x_ratio = finger_pos[0] / transformed_frame.shape[1]
                    y_ratio = finger_pos[1] / transformed_frame.shape[0]
                    pyautogui.moveTo(x_ratio * screen_width, y_ratio * screen_height)
                    print(f"Cursor moved to: {x_ratio * screen_width}, {y_ratio * screen_height}")

                cv2.drawContours(frame, [touchpad], -1, (0, 255, 0), 2)

            cv2.imshow("Virtual Touchpad", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

if __name__ == "__main__":
    main()