import math
import time
import cv2
import mediapipe as mp
import numpy as np

_head_direction = None
_direction_start_time = None
_alert_active = False
_alert_start_time = None

YAW_THRESHOLD = 15
PITCH_THRESHOLD = 15
ALERT_DURATION = 2.0
ALERT_DISPLAY_TIME = 3.0

def _get_angles(rotation_matrix):
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

def _get_direction(pitch, yaw):
    if abs(yaw) > YAW_THRESHOLD:
        return "Left" if yaw > 0 else "Right"
    elif abs(pitch) > PITCH_THRESHOLD:
        return "Down" if pitch > 0 else "Up"
    else:
        return "Center"

def process_head_pose(image, face_mesh, current_time):
    global _head_direction, _direction_start_time, _alert_active, _alert_start_time

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    h, w, _ = image.shape
    
    pitch, yaw, roll = 0, 0, 0
    current_direction = "Center"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            model_points = np.array([
                [285, 528, 200], [285, 371, 152], [197, 574, 128],
                [173, 425, 108], [360, 574, 128], [391, 425, 108]
            ], dtype=np.float64)
            
            image_points = []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    image_points.append([x, y])
            
            if len(image_points) == 6:
                image_points = np.array(image_points, dtype=np.float64)
                
                focal_length = w
                camera_matrix = np.array([[focal_length, 0, w/2],
                                          [0, focal_length, h/2],
                                          [0, 0, 1]])
                
                dist_coeffs = np.zeros((4, 1))
                
                success_pnp, rotation_vec, translation_vec = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs)
                
                if success_pnp:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                    pitch, yaw, roll = _get_angles(rotation_matrix)
                    
                    cv2.putText(image, f'Pitch: {int(pitch)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                    cv2.putText(image, f'Yaw: {int(yaw)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                    cv2.putText(image, f'Roll: {int(roll)}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
                    
                    current_direction = _get_direction(pitch, yaw)
                    cv2.putText(image, f'Direction: {current_direction}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if current_direction != "Center":
                        if _head_direction != current_direction:
                            _head_direction = current_direction
                            _direction_start_time = current_time
                            _alert_active = False
                        else:
                            if _direction_start_time is not None:
                                duration = current_time - _direction_start_time
                                remaining = max(0, ALERT_DURATION - duration)
                                cv2.putText(image, f'Time: {remaining:.1f}s', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                
                                if duration >= ALERT_DURATION and not _alert_active:
                                    _alert_active = True
                                    _alert_start_time = current_time
                    else:
                        _head_direction = None
                        _direction_start_time = None
                        _alert_active = False
    
    if _alert_active and _alert_start_time is not None:
        alert_duration = current_time - _alert_start_time
        if alert_duration < ALERT_DISPLAY_TIME:
            cv2.putText(image, "LOOK AWAY!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        else:
            _alert_active = False
            _alert_start_time = None

    return image, {"pitch": pitch, "yaw": yaw, "roll": roll, "direction": current_direction, "alert_active": _alert_active}
