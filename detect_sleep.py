import math
import numpy as np

# ==============================================================================
# Section 1: Module Settings and Constants
# ==============================================================================
EAR_THRESHOLD = 0.22  
DROWSY_TIME_SECONDS = 1  
MAR_THRESHOLD = 0.5   
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [13, 14, 78, 308] # Top, Bottom, Right, Left inner lips

# ==============================================================================
# Section 2: State Variables
# ==============================================================================
_sleep_start_time = 0
# ==============================================================================
# Section 3: Helper Calculation Functions
# ==============================================================================
def _calculate_euclidean_distance(p1, p2):
    """Calculates the straight-line distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _calculate_eye_aspect_ratio(landmarks, eye_indices):
    try:
        vertical_dist_1 = _calculate_euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        vertical_dist_2 = _calculate_euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        
        horizontal_dist = _calculate_euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        
        # Calculate EAR, handle division by zero
        ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
        return ear if horizontal_dist > 0 else 0.0
    except IndexError:
        return 0.0

def _calculate_mouth_aspect_ratio(landmarks):
    try:
        # Vertical and horizontal mouth landmarks
        vertical_dist = _calculate_euclidean_distance(landmarks[MOUTH_INDICES[0]], landmarks[MOUTH_INDICES[1]])
        horizontal_dist = _calculate_euclidean_distance(landmarks[MOUTH_INDICES[2]], landmarks[MOUTH_INDICES[3]])
        
        # Calculate MAR, handle division by zero
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.0
    except IndexError:
        # Return 0.0 if any landmark index is out of bounds
        return 0.0

# ==============================================================================
# Section 4: Main Processing Function
# ==============================================================================
def process_sleep_detection(face_mesh_results, frame_shape, current_time):
    global _sleep_start_time
    
    # Initialize default return values
    sleep_alert = False
    yawn_alert = False

    if not (face_mesh_results and face_mesh_results.multi_face_landmarks):
        # If no face is detected, return default values immediately
        return {"sleep_alert": sleep_alert, "yawn_alert": yawn_alert}

    # --- Data Extraction ---
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    h, w, _ = frame_shape
    # Convert normalized landmarks to pixel coordinates
    landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

    # --- Calculations ---
    left_ear = _calculate_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = _calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
    average_ear = (left_ear + right_ear) / 2.0
    mar = _calculate_mouth_aspect_ratio(landmarks)

    # --- Drowsiness Detection Logic ---
    if average_ear < EAR_THRESHOLD:
        # If eyes are closed, check if this is the start of a closure
        if _sleep_start_time == 0:
            _sleep_start_time = current_time
        # If eyes have been closed for long enough, trigger the alert
        elif (current_time - _sleep_start_time) >= DROWSY_TIME_SECONDS:
            sleep_alert = True
    else:
        # If eyes are open, reset the timer
        _sleep_start_time = 0

    # --- Yawn Detection Logic ---
    if mar > MAR_THRESHOLD:
        yawn_alert = True

    # --- Return Results ---
    return {"sleep_alert": sleep_alert, "yawn_alert": yawn_alert}
