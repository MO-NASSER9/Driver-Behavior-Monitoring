import cv2
import numpy as np
import time

class EyeGaze:
    def __init__(self, distraction_threshold_seconds=1.0):
        self.is_distracted = False
        self.distraction_start_time = 0
        self.distraction_threshold = distraction_threshold_seconds
        self.last_alert_time = 0
        self.alert_cooldown = 3.0

    def _get_gaze_direction(self, frame, face_landmarks):
        img_h, img_w = frame.shape[:2]
        landmarks_2d = np.array([(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark])

        try:
            left_iris_coords = tuple(landmarks_2d[473].astype(int))
            right_iris_coords = tuple(landmarks_2d[468].astype(int))

            l_eye_right_corner = landmarks_2d[33]
            l_eye_left_corner = landmarks_2d[133]
            r_eye_right_corner = landmarks_2d[263]
            r_eye_left_corner = landmarks_2d[362]

            l_eye_width = np.linalg.norm(l_eye_right_corner - l_eye_left_corner)
            r_eye_width = np.linalg.norm(r_eye_right_corner - r_eye_left_corner)

            if l_eye_width < 5 or r_eye_width < 5:
                return "Center", None, None

            l_iris_pos_ratio = (left_iris_coords[0] - l_eye_left_corner[0]) / l_eye_width
            r_iris_pos_ratio = (right_iris_coords[0] - r_eye_left_corner[0]) / r_eye_width

            gaze_thresh_low = 0.01
            gaze_thresh_high = 0.99
            
            left_eye_off_center = l_iris_pos_ratio < gaze_thresh_low or l_iris_pos_ratio > gaze_thresh_high
            right_eye_off_center = r_iris_pos_ratio < gaze_thresh_low or r_iris_pos_ratio > gaze_thresh_high

            if left_eye_off_center and right_eye_off_center:
                final_direction = "Not Center"
            else:
                final_direction = "Center"
                
            return final_direction, left_iris_coords, right_iris_coords

        except (IndexError, TypeError):
            return "Center", None, None

    def process_frame(self, frame, face_mesh_results):
        gaze_alert = None
        left_iris = None
        right_iris = None
        current_time = time.time()

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            direction, left_iris, right_iris = self._get_gaze_direction(frame, face_landmarks)
            
            if direction == "Not Center":
                if not self.is_distracted:
                    self.is_distracted = True
                    self.distraction_start_time = current_time
            else:
                self.is_distracted = False
                self.distraction_start_time = 0

            if self.is_distracted:
                distraction_duration = current_time - self.distraction_start_time
                if distraction_duration > self.distraction_threshold and (current_time - self.last_alert_time > self.alert_cooldown):
                    gaze_alert = "PLEASE FOCUS!"
                    self.last_alert_time = current_time
                    self.is_distracted = False 
        else:
            self.is_distracted = False
            self.distraction_start_time = 0

        return gaze_alert, left_iris, right_iris

    def close(self):
        pass
