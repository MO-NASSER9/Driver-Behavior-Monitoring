import cv2
import numpy as np
import time
import json
import os

class EyeGaze:
    # ==============================================================================
    # Section 1: Initialization and Settings
    # ==============================================================================
    def __init__(self, distraction_threshold_seconds=1.0, calibration_time=3, sensitivity=0.07, calibration_file="gaze_calibration.json"):
        self.distraction_threshold = distraction_threshold_seconds 
        self.alert_cooldown = 2.0 
        self.calibration_frames = int(calibration_time * 30)  
        self.sensitivity = sensitivity  
        self.is_calibrated = False 
        self.distraction_start_time = 0  
        self.last_alert_time = 0  
        self.calibration_data = [] 
        self.baseline_gaze_ratio = 0.0
        self.calibration_file = calibration_file

        # --- Landmark Indices ---
        self.left_eye_indices = [33, 133, 473]  # [Right Corner, Left Corner, Iris]
        self.right_eye_indices = [263, 362, 468] # [Right Corner, Left Corner, Iris]

        # --- Load calibration if available ---
        self._load_calibration()

    # ==============================================================================
    # Section 2: Calibration Handling
    # ==============================================================================
    def _load_calibration(self):
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, "r") as f:
                data = json.load(f)
                self.baseline_gaze_ratio = data.get("baseline_gaze_ratio", 0.0)
                self.is_calibrated = True
                print(f"[INFO] Loaded calibration: Baseline = {self.baseline_gaze_ratio:.3f}")

    def _save_calibration(self):
        with open(self.calibration_file, "w") as f:
            json.dump({"baseline_gaze_ratio": self.baseline_gaze_ratio}, f)
        print(f"[INFO] Calibration saved to {self.calibration_file}")

    # ==============================================================================
    # Section 3: Helper Calculation Function
    # ==============================================================================
    def _calculate_gaze_ratio(self, landmarks, eye_indices):
        try:
            eye_right_pt = landmarks[eye_indices[0]]
            eye_left_pt = landmarks[eye_indices[1]]
            iris_pt = landmarks[eye_indices[2]]

            eye_center_x = (eye_right_pt[0] + eye_left_pt[0]) / 2
            eye_width = np.linalg.norm(eye_right_pt - eye_left_pt)

            if eye_width == 0:
                return None

            return (iris_pt[0] - eye_center_x) / eye_width
        except IndexError:
            return None

    # ==============================================================================
    # Section 4: Main Processing Function
    # ==============================================================================
    def process_frame(self, frame, face_mesh_results):
        gaze_alert = None
        left_iris_coords = None
        right_iris_coords = None

        if not (face_mesh_results and face_mesh_results.multi_face_landmarks):
            return None, None, None

        # --- تحويل نقاط الوجه إلى مصفوفة بكسلات ---
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

        try:
            left_iris_coords = tuple(landmarks[self.left_eye_indices[2]].astype(int))
            right_iris_coords = tuple(landmarks[self.right_eye_indices[2]].astype(int))
        except IndexError:
            pass

        # --- 4.1: Calibration Phase ---
        if not self.is_calibrated:
            cv2.putText(frame, "Calibrating Gaze... Look Forward", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            left_ratio = self._calculate_gaze_ratio(landmarks, self.left_eye_indices)
            right_ratio = self._calculate_gaze_ratio(landmarks, self.right_eye_indices)

            if left_ratio is not None and right_ratio is not None:
                avg_ratio = (left_ratio + right_ratio) / 2.0
                self.calibration_data.append(avg_ratio)

            if len(self.calibration_data) >= self.calibration_frames:
                self.baseline_gaze_ratio = np.mean(self.calibration_data)
                self.is_calibrated = True
                self._save_calibration()
                print(f"[INFO] Gaze calibration complete. Baseline: {self.baseline_gaze_ratio:.3f}")

            return None, left_iris_coords, right_iris_coords

        # --- 4.2: Detection Phase (Post-Calibration) ---
        current_time = time.time()
        left_gaze_ratio = self._calculate_gaze_ratio(landmarks, self.left_eye_indices)
        right_gaze_ratio = self._calculate_gaze_ratio(landmarks, self.right_eye_indices)

        if left_gaze_ratio is None or right_gaze_ratio is None:
            return None, left_iris_coords, right_iris_coords

        avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0

        if abs(avg_gaze_ratio - self.baseline_gaze_ratio) > self.sensitivity:
            if self.distraction_start_time == 0:
                self.distraction_start_time = current_time
            elif (current_time - self.distraction_start_time) > self.distraction_threshold:
                if (current_time - self.last_alert_time) > self.alert_cooldown:
                    gaze_alert = "GAZE DISTRACTION!"
                    self.last_alert_time = current_time
        else:
            self.distraction_start_time = 0

        return gaze_alert, left_iris_coords, right_iris_coords
