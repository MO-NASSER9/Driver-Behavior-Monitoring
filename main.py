import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
import detect_sleep
import head_pose
from eye_gaze import EyeGaze

YOLO_MODEL_PATH = "/media/mo/Mohamed/ME/Coruses/A I/summit/project/monitoring-driver/best.pt"
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
ALERT_DURATION_SECONDS = 1

try:
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(f"Please check if the file exists at this path: {YOLO_MODEL_PATH}")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
eye_gaze_tracker = EyeGaze(distraction_threshold_seconds=1.0)

def open_camera(width, height):
    for i in range(5, -1, -1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
    return None

cap = open_camera(TARGET_WIDTH, TARGET_HEIGHT)
if not cap:
    print("No available camera found. Exiting...")
    exit()

active_alerts = {}

def add_alert_if_not_present(key, message, current_time):
    if key not in active_alerts:
        active_alerts[key] = {"message": message, "time": current_time}

while True:
    success, frame = cap.read()
    if not success:
        time.sleep(0.1)
        continue
    
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    current_time = time.time()

    yolo_results = model.predict(frame, conf=0.25, verbose=False)
    display_frame = yolo_results[0].plot()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    sleep_info = detect_sleep.process_sleep_detection(results, display_frame.shape, current_time)
    head_pose_info = head_pose.process_head_pose(results, display_frame.shape, current_time)
    gaze_alert, left_iris_coords, right_iris_coords = eye_gaze_tracker.process_frame(display_frame, results)

    if left_iris_coords: 
        cv2.circle(display_frame, left_iris_coords, radius=3, color=(255, 255, 255), thickness=-1)
    if right_iris_coords: 
        cv2.circle(display_frame, right_iris_coords, radius=3, color=(255, 255, 255), thickness=-1)

    if sleep_info.get("sleep_alert"): add_alert_if_not_present("sleep", "Driver is sleepy!", current_time)
    if sleep_info.get("yawn_alert"): add_alert_if_not_present("yawn", "Yawning detected!", current_time)
    if head_pose_info.get("alert_active"): add_alert_if_not_present("head_pose", "LOOKING AWAY!", current_time)
    if gaze_alert: add_alert_if_not_present("gaze", gaze_alert, current_time)

    pitch = head_pose_info.get('pitch', 0)
    yaw = head_pose_info.get('yaw', 0)
    roll = head_pose_info.get('roll', 0)
    
    cv2.putText(display_frame, f'Pitch: {int(pitch)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
    cv2.putText(display_frame, f'Yaw: {int(yaw)}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
    cv2.putText(display_frame, f'Roll: {int(roll)}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

    alerts_to_remove = [key for key, data in active_alerts.items() if current_time - data["time"] >= ALERT_DURATION_SECONDS]
    for key in alerts_to_remove: 
        del active_alerts[key]

    y_pos = 130
    for data in active_alerts.values():
        cv2.putText(display_frame, data["message"], (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_pos += 40

    cv2.imshow("Driver Behavior Monitor", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
