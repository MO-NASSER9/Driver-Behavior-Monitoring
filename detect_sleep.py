import math
import numpy as np

_sleep_start = None
EAR_THRESHOLD = 0.25
DROWSY_TIME = 2
MAR_THRESHOLD = 0.6
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def _eye_aspect_ratio(landmarks, eye_indices):
    if not all(idx < len(landmarks) for idx in eye_indices):
        return 0.0
    A = _euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = _euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = _euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

def _mouth_aspect_ratio(landmarks):
    try:
        if not all(idx < len(landmarks) for idx in [13, 14, 78, 308]):
            return 0.0
        A = _euclidean_distance(landmarks[13], landmarks[14])
        B = _euclidean_distance(landmarks[78], landmarks[308])
        if B == 0: return 0.0
        return A / B
    except IndexError:
        return 0.0

def process_sleep_detection(face_mesh_results, frame_shape, current_time):
    global _sleep_start
    
    h, w, _ = frame_shape
    sleep_alert = False
    yawn_alert = False

    if face_mesh_results and face_mesh_results.multi_face_landmarks:
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        landmarks = np.array([(int(lm.x*w), int(lm.y*h)) for lm in face_landmarks.landmark])

        left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = _mouth_aspect_ratio(landmarks)

        if ear < EAR_THRESHOLD:
            if _sleep_start is None:
                _sleep_start = current_time
            elif current_time - _sleep_start >= DROWSY_TIME:
                sleep_alert = True
        else:
            _sleep_start = None

        if mar > MAR_THRESHOLD:
            yawn_alert = True

    return {"sleep_alert": sleep_alert, "yawn_alert": yawn_alert}
