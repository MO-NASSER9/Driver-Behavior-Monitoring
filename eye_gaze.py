import cv2
import numpy as np
import time

class EyeGaze:
    def __init__(self, distraction_threshold_seconds=1.5, calibration_time=5, sensitivity=0.07):
        self.distraction_threshold = distraction_threshold_seconds
        self.alert_cooldown = 1
        self.calibration_frames = int(calibration_time * 30)
        self.sensitivity = sensitivity
        self.is_calibrated = False
        self.is_distracted = False
        self.distraction_start_time = 0
        self.last_alert_time = 0
        self.calibration_data = []
        self.baseline_gaze_ratio = 0.0

    def _calculate_gaze_ratio(self, eye_points, facial_landmarks):
        try:
            eye_center_x = (facial_landmarks[eye_points[0]][0] + facial_landmarks[eye_points[3]][0]) / 2
            eye_center_y = (facial_landmarks[eye_points[1]][1] + facial_landmarks[eye_points[4]][1]) / 2
            iris_pos_x = facial_landmarks[eye_points[6]][0]
            eye_width = np.linalg.norm(facial_landmarks[eye_points[0]] - facial_landmarks[eye_points[3]])
            
            if eye_width == 0:
                return None
            
            ratio = (iris_pos_x - eye_center_x) / eye_width
            return ratio
        except IndexError:
            return None

    def process_frame(self, frame, face_mesh_results):
        gaze_alert = None
        left_iris_coords = None
        right_iris_coords = None
        
        if not face_mesh_results or not face_mesh_results.multi_face_landmarks:
            return None, None, None

        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

        left_eye_points = [33, 159, 160, 133, 145, 153, 473]
        right_eye_points = [263, 386, 387, 362, 374, 380, 468]

        left_gaze_ratio = self._calculate_gaze_ratio(left_eye_points, landmarks)
        right_gaze_ratio = self._calculate_gaze_ratio(right_eye_points, landmarks)

        try:
            left_iris_coords = tuple(landmarks[left_eye_points[6]].astype(int))
            right_iris_coords = tuple(landmarks[right_eye_points[6]].astype(int))
        except IndexError:
            pass

        if left_gaze_ratio is None or right_gaze_ratio is None:
            return None, left_iris_coords, right_iris_coords

        avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0

        if not self.is_calibrated:
            if len(self.calibration_data) < self.calibration_frames:
                self.calibration_data.append(avg_gaze_ratio)
            else:
                self.baseline_gaze_ratio = np.mean(self.calibration_data)
                self.is_calibrated = True
                print(f"Gaze calibration complete. Baseline ratio: {self.baseline_gaze_ratio:.3f}")
            
            cv2.putText(frame, "Calibrating... Look Forward", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return None, left_iris_coords, right_iris_coords

        current_time = time.time()
        
        if abs(avg_gaze_ratio - self.baseline_gaze_ratio) > self.sensitivity:
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

        return gaze_alert, left_iris_coords, right_iris_coords
