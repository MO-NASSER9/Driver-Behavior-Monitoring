# Real-Time Driver Behavior Monitoring System

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/Frameworks-OpenCV%20%7C%20MediaPipe%20%7C%20YOLOv8-orange" alt="Frameworks">
    <img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version">
</p>

## 1. Overview

This project is a real-time driver behavior monitoring system developed in Python. It leverages computer vision and deep learning techniques to detect signs of drowsiness, distraction, and other potentially hazardous activities. The system processes a live video feed from a standard webcam to provide timely alerts, aiming to enhance driver safety and prevent accidents.

## 2. Features

*   **Drowsiness Detection**: Monitors the driver's Eye Aspect Ratio (EAR ).
*   **Yawn Detection**: Monitors the Mouth Aspect Ratio (MAR).
*   **Gaze Tracking**: Tracks iris movement to detect distraction.
*   **Head Pose Estimation**: Detects if the driver's head is turned away.
*   **Object-Based Distraction Detection**: Utilizes a YOLOv8 model to detect objects like mobile phones.

## 3. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MO-NASSER9/Driver-Behavior-Monitoring
    cd Mohamed-Nasser
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model:**
    -   Download the pre-trained `best.pt` model file and place it in the root directory.

## 4. How to Run

```bash
python main.py
```

**Current Version:** `1.0.0`
