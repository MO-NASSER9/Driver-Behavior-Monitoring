import numpy as np
# --- 1. Module Settings ---
YAW_RATIO_THRESHOLD = 0.2
PITCH_RATIO_THRESHOLD = 0.15
ALERT_DURATION = 1.0
ALERT_COOLDOWN = 2.0

# --- 2. State Variables ---
_current_direction = "Forward"
_direction_start_time = 0
_last_alert_time = 0

# --- 3. Facial Landmark Indices ---
NOSE_TIP = 1
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
FOREHEAD_CENTER = 10
CHIN_BOTTOM = 152

# --- 4. Main Processing Function ---
def process_head_pose(face_mesh_results, frame_shape, current_time):
    global _current_direction, _direction_start_time, _last_alert_time
    alert_active = False
    alert_direction = None  
    new_direction = "Forward"

    if face_mesh_results and face_mesh_results.multi_face_landmarks:
        face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
        h, w, _ = frame_shape

        try:
            nose = (face_landmarks[NOSE_TIP].x * w, face_landmarks[NOSE_TIP].y * h)
            left_eye = (face_landmarks[LEFT_EYE_CORNER].x * w, face_landmarks[LEFT_EYE_CORNER].y * h)
            right_eye = (face_landmarks[RIGHT_EYE_CORNER].x * w, face_landmarks[RIGHT_EYE_CORNER].y * h)
            forehead = (face_landmarks[FOREHEAD_CENTER].x * w, face_landmarks[FOREHEAD_CENTER].y * h)
            chin = (face_landmarks[CHIN_BOTTOM].x * w, face_landmarks[CHIN_BOTTOM].y * h)

            eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            yaw_ratio = (nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_dist

            face_height = np.linalg.norm(np.array(forehead) - np.array(chin))
            pitch_ratio = (nose[1] - (forehead[1] + chin[1]) / 2) / face_height

            if yaw_ratio > YAW_RATIO_THRESHOLD: new_direction = "Right"
            elif yaw_ratio < -YAW_RATIO_THRESHOLD: new_direction = "Left"
            elif pitch_ratio > PITCH_RATIO_THRESHOLD: new_direction = "Down"
            elif pitch_ratio < -PITCH_RATIO_THRESHOLD: new_direction = "Up"
            else: new_direction = "Forward"

        except (IndexError, ZeroDivisionError):
            new_direction = "Forward"

    # --- 5. Alert Logic ---
    if new_direction != "Forward":
        if _current_direction != new_direction:
            _current_direction = new_direction
            _direction_start_time = current_time
        elif (current_time - _direction_start_time) > ALERT_DURATION:
            if (current_time - _last_alert_time) > ALERT_COOLDOWN:
                alert_active = True
                alert_direction = _current_direction 
                _last_alert_time = current_time
    else:
        _current_direction = "Forward"
        _direction_start_time = 0

    # --- 6. Return Dictionary ---
    return {
        "alert_active": alert_active,
        "alert_direction": alert_direction,
        "current_direction": _current_direction
    }
