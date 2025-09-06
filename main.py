# ==============================================================================
# Project: Driver Behavior Monitoring System
# Author: Mohamed Nasser Abdelhady
# Version: 1.0.0
# Date: 2025-09-06
# Description: Real-time driver monitoring using YOLO, Mediapipe, and Gaze Tracking.
# ==============================================================================

VERSION = "1.0.0"

# ==============================================================================
# Section 1: Core Library Imports
# ==============================================================================
import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import head_pose
import detect_sleep
import eye_gaze

# ==============================================================================
# Section 2: Configuration and Constants
# ==============================================================================
YOLO_MODEL_PATH = "best.pt"
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALERT_DISPLAY_DURATION = 2.0

# ==============================================================================
# Section 3: Initialization Functions
# ==============================================================================
def load_yolo_model(path):
    try:
        model = YOLO(path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"FATAL: Error loading YOLO model: {e}")
        return None

def initialize_camera():
    for camera_index in range(2, -1, -1):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            if cap.read()[0]:
                print(f"Camera found and working at index: {camera_index}")
                return cap
            else:
                cap.release()
    print("FATAL: No working camera found. Exiting.")
    return None

# ==============================================================================
# Section 4: Main Program Execution
# ==============================================================================
if __name__ == "__main__":
    
    # --- 4.1: Initialize Objects and Models ---
    cap = initialize_camera()
    if not cap: exit()

    model = load_yolo_model(YOLO_MODEL_PATH)
    if not model: exit()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    eye_gaze_tracker = eye_gaze.EyeGaze(calibration_time=3)
    active_alerts = {}
    print(f"Starting Driver Monitor System - Version {VERSION}")
    print("\nStarting Driver Monitor...")

    # --- 4.2: Main Video Processing Loop ---
    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: Lost connection to camera. Retrying...")
            time.sleep(1)
            continue
        
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame = cv2.flip(frame, 1)
        current_time = time.time()

        yolo_results = model.predict(frame, conf=0.4, verbose=False, half=True)
        display_frame = yolo_results[0].plot()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)

        # --- 4.3: Manage Alerts and Apply Prioritization Logic ---
        expired_keys = [k for k, data in active_alerts.items() if current_time - data["time"] >= ALERT_DISPLAY_DURATION]
        for k in expired_keys:
            if k in active_alerts:
                del active_alerts[k]

        head_pose_info = head_pose.process_head_pose(face_results, display_frame.shape, current_time)
        head_alert_active = head_pose_info.get("alert_active")
        head_alert_direction = head_pose_info.get("alert_direction")
        head_direction = head_pose_info.get("current_direction", "Forward") # We need the current direction

        run_secondary_detectors = (head_direction == "Forward")

        if not run_secondary_detectors:
            # Priority for head pose alerts
            if head_alert_active and head_alert_direction:
                alert_message_l1 = f"Looking {head_alert_direction}"
                alert_message_l2 = "Please Look Forward"
                active_alerts["head_pose"] = {"message_line1": alert_message_l1, "message_line2": alert_message_l2, "time": current_time}
            
            # Delete any other potentially conflicting alerts
            for key in ["sleep", "yawn", "gaze"]:
                if key in active_alerts:
                    del active_alerts[key]
        else: 
            # Only if head is forward, check for other alerts
            if "head_pose" in active_alerts:
                del active_alerts["head_pose"]

            if face_results and face_results.multi_face_landmarks:
                sleep_info = detect_sleep.process_sleep_detection(face_results, display_frame.shape, current_time)
                gaze_alert, left_iris, right_iris = eye_gaze_tracker.process_frame(display_frame, face_results)
                
                if sleep_info.get("sleep_alert"): 
                    active_alerts["sleep"] = {"message_line1": "SLEEPY!", "message_line2": "", "time": current_time}
                if sleep_info.get("yawn_alert"): 
                    active_alerts["yawn"] = {"message_line1": "YAWNING!", "message_line2": "", "time": current_time}
                if gaze_alert: 
                    active_alerts["gaze"] = {"message_line1": gaze_alert, "message_line2": "", "time": current_time}

        # --- 4.4: Draw Overlays on the Display Frame ---
        if 'left_iris' in locals() and left_iris:
            cv2.circle(display_frame, left_iris, 3, (255, 255, 255), -1)
        if 'right_iris' in locals() and right_iris:
            cv2.circle(display_frame, right_iris, 3, (255, 255, 255), -1)

        y_pos = 30
        for data in list(active_alerts.values()):
            cv2.putText(display_frame, data["message_line1"], (20, y_pos), FONT, 1, (0, 0, 255), 2)
            if data.get("message_line2"):
                cv2.putText(display_frame, data["message_line2"], (20, y_pos + 30), FONT, 0.7, (0, 165, 255), 2)
            y_pos += 60

        # --- 4.5: Print Debug Info to the Terminal ---
        print("="*30)
        print(f"Timestamp: {current_time:.2f} | Head Pose: {head_direction}")
        
        if run_secondary_detectors:
            is_sleepy = "Yes" if "sleep" in active_alerts else "No"
            is_yawning = "Yes" if "yawn" in active_alerts else "No"
            print(f"Sleepy: {is_sleepy} | Yawning: {is_yawning}")
        else:
            print("Eye/Sleep detectors paused (Head is turned)")

        if active_alerts:
            print(f"Active Alerts on Screen: {list(active_alerts.keys())}")
        else:
            print("Active Alerts on Screen: None")

        # --- 4.6: Display the Frame and Wait for Exit Command ---
        cv2.imshow("Driver Behavior Monitor", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # --- 4.7: Cleanup and Shutdown ---
    print("\nClosing application...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")
