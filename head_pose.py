import math
import time
import cv2
import numpy as np

YAW_THRESHOLD = 20
PITCH_THRESHOLD = 15
ALERT_DURATION = 1.5
ALERT_DISPLAY_TIME = 3.0

_head_direction = None
_direction_start_time = None
_alert_active = False
_alert_start_time = None

def _calculate_head_angles(landmarks, frame_width, frame_height):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    image_points = np.array([
        landmarks[1], landmarks[152], landmarks[263],
        landmarks[33], landmarks[287], landmarks[57]
    ], dtype=np.float64)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, _) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0, 0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    pitch = math.degrees(math.asin(rotation_matrix[2, 0]))
    yaw = math.degrees(-math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    roll = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    
    return pitch, yaw, roll

def _get_direction(pitch, yaw):
    if yaw > YAW_THRESHOLD: return "Right"
    elif yaw < -YAW_THRESHOLD: return "Left"
    elif pitch > PITCH_THRESHOLD: return "Down"
    elif pitch < -PITCH_THRESHOLD: return "Up"
    else: return "Center"

def process_head_pose(face_mesh_results, frame_shape, current_time):
    global _head_direction, _direction_start_time, _alert_active, _alert_start_time

    h, w, _ = frame_shape
    direction = "Center"
    alert_now = False
    pitch, yaw, roll = 0, 0, 0

    if face_mesh_results and face_mesh_results.multi_face_landmarks:
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
        
        pitch, yaw, roll = _calculate_head_angles(landmarks, w, h)
        direction = _get_direction(pitch, yaw)

        if direction != "Center":
            if _head_direction != direction:
                _head_direction = direction
                _direction_start_time = current_time
                _alert_active = False
            elif _direction_start_time is not None:
                duration = current_time - _direction_start_time
                if duration >= ALERT_DURATION and not _alert_active:
                    _alert_active = True
                    _alert_start_time = current_time
        else:
            _head_direction = None
            _direction_start_time = None
            _alert_active = False

    if _alert_active and _alert_start_time is not None:
        alert_now = True
        if (current_time - _alert_start_time) >= ALERT_DISPLAY_TIME:
            _alert_active = False
            _alert_start_time = None
            alert_now = False
            
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "direction": direction, "alert_active": alert_now}
