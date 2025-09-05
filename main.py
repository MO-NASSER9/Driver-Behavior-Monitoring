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
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_yolo_model(path):
    try:
        model = YOLO(path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print(f"Please check if the file exists at this path: {path}")
        return None

def initialize_camera():
    for i in range(5, -1, -1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found and opened at index: {i}")
            return cap
    print("No available camera found. Exiting...")
    return None

model = load_yolo_model(YOLO_MODEL_PATH)
if not model:
    exit()

cap = initialize_camera()
if not cap:
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
eye_gaze_tracker = EyeGaze(distraction_threshold_seconds=1.0)

active_alerts = {}

def add_alert(key, message, current_time):
    if key not in active_alerts:
        active_alerts[key] = {"message": message, "time": current_time}

def draw_iris_circles(frame, left_coords, right_coords):
    if left_coords:
        cv2.circle(frame, left_coords, radius=3, color=(255, 255, 255), thickness=-1)
    if right_coords:
        cv2.circle(frame, right_coords, radius=3, color=(255, 255, 255), thickness=-1)

def draw_head_pose_info(frame, pitch, yaw, roll):
    cv2.putText(frame, f'Pitch: {int(pitch)}', (20, 30), FONT, 0.7, (200, 0, 200), 2)
    cv2.putText(frame, f'Yaw: {int(yaw)}', (20, 60), FONT, 0.7, (200, 0, 200), 2)
    cv2.putText(frame, f'Roll: {int(roll)}', (20, 90), FONT, 0.7, (200, 0, 200), 2)

def display_active_alerts(frame):
    y_pos = 130
    for data in active_alerts.values():
        cv2.putText(frame, data["message"], (20, y_pos), FONT, 1, (0, 0, 255), 2)
        y_pos += 40

while True:
    success, frame = cap.read()
    if not success:
        time.sleep(0.1)
        continue

    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    current_time = time.time()

    yolo_results = model.predict(frame, conf=0.25, verbose=False)
    display_frame = yolo_results[0].plot()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)

    sleep_info = detect_sleep.process_sleep_detection(face_results, display_frame.shape, current_time)
    head_pose_info = head_pose.process_head_pose(face_results, display_frame.shape, current_time)
    gaze_alert, left_iris, right_iris = eye_gaze_tracker.process_frame(display_frame, face_results)

    if sleep_info.get("sleep_alert"): add_alert("sleep", "Driver is sleepy!", current_time)
    if sleep_info.get("yawn_alert"): add_alert("yawn", "Yawning detected!", current_time)
    if head_pose_info.get("alert_active"): add_alert("head_pose", "LOOKING AWAY!", current_time)
    if gaze_alert: add_alert("gaze", gaze_alert, current_time)

    expired_keys = [key for key, data in active_alerts.items() if current_time - data["time"] >= ALERT_DURATION_SECONDS]
    for key in expired_keys:
        del active_alerts[key]

    draw_iris_circles(display_frame, left_iris, right_iris)
    draw_head_pose_info(display_frame, head_pose_info.get('pitch', 0), head_pose_info.get('yaw', 0), head_pose_info.get('roll', 0))
    display_active_alerts(display_frame)
    
    print("="*30)
    print(f"Timestamp: {current_time:.2f}")
    print(f"Head Pose (Pitch, Yaw, Roll): {head_pose_info.get('pitch', 0):.1f}, {head_pose_info.get('yaw', 0):.1f}, {head_pose_info.get('roll', 0):.1f}")
    
    is_sleepy = "Yes" if sleep_info.get("sleep_alert") else "No"
    is_yawning = "Yes" if sleep_info.get("yawn_alert") else "No"
    print(f"Sleepy: {is_sleepy}, Yawning: {is_yawning}")

    if active_alerts:
        alert_messages = [data["message"] for data in active_alerts.values()]
        print(f"Active Alerts: {alert_messages}")
    else:
        print("Active Alerts: None")

    cv2.imshow("Driver Behavior Monitor", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.")
