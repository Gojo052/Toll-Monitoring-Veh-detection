import cv2
import numpy as np
import time
from datetime import datetime
import os
import json
from ultralytics import YOLO  # YOLOv8 (pip install ultralytics)
import easyocr  # Lightweight OCR (pip install easyocr)

# ===== CONFIGURATION =====
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "rtsp_url": "rtsp://admin:ONPL$LRR$2024@192.168.50.73",
    "resolution": [1280, 720],
    "lane_roi": None,
    "printer_roi": None,
    "detection_threshold": 0.4,
    "white_threshold": 200,
    "min_receipt_area": 500,
    "check_interval": 0.5,
    "log_path": "logs"
}

# ===== SYSTEM INITIALIZATION =====
class TollMonitor:
    def __init__(self):
        self.load_config()
        self.setup_directories()
        self.init_models()
        self.cap = self.init_camera()
        self.vehicle_detected = False
        self.last_detection_time = 0
        self.current_status = "Waiting for vehicle"

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
            print("Loaded existing configuration")
        except Exception as e:
            print(f"Error loading config: {e}. Creating new config.")
            self.config = DEFAULT_CONFIG
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)

    def setup_directories(self):
        os.makedirs(self.config["log_path"], exist_ok=True)
        os.makedirs(os.path.join(self.config["log_path"], "snapshots"), exist_ok=True)

    def init_models(self):
        self.vehicle_detector = YOLO('yolov8n.pt')
        self.ocr = easyocr.Reader(['en'], gpu=False)

    def init_camera(self):
        cap = cv2.VideoCapture(self.config["rtsp_url"])
        if not cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["resolution"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["resolution"][1])
        return cap

    # ===== ROI MANAGEMENT =====
    def draw_rois(self, frame):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.config["lane_roi"] is None:
                    self.config["lane_roi"] = [(x, y)]
                elif len(self.config["lane_roi"]) == 1:
                    self.config["lane_roi"].append((x, y))
                    print("Lane ROI set")
                elif self.config["printer_roi"] is None:
                    self.config["printer_roi"] = [(x, y)]
                elif len(self.config["printer_roi"]) == 1:
                    self.config["printer_roi"].append((x, y))
                    print("Printer ROI set")
                    self.save_config()

        clone = frame.copy()
        cv2.namedWindow("ROI Setup")
        cv2.setMouseCallback("ROI Setup", mouse_callback)

        while True:
            display = clone.copy()
            if self.config["lane_roi"] and len(self.config["lane_roi"]) == 2:
                cv2.line(display, *self.config["lane_roi"], (0, 0, 255), 2)
            if self.config["printer_roi"] and len(self.config["printer_roi"]) == 2:
                cv2.rectangle(display, *self.config["printer_roi"], (0, 255, 0), 2)

            cv2.imshow("ROI Setup", display)
            key = cv2.waitKey(1)
            if key == 27 or (self.config["lane_roi"] and len(self.config["lane_roi"]) == 2 and self.config["printer_roi"] and len(self.config["printer_roi"]) == 2):
                break

        cv2.destroyWindow("ROI Setup")

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    # ===== DETECTION LOGIC =====
    def detect_vehicle(self, frame):
        if not self.config["lane_roi"] or len(self.config["lane_roi"]) != 2:
            return False

        roi = self.get_roi_mask(frame, self.config["lane_roi"])
        results = self.vehicle_detector(roi, verbose=False)

        for box in results[0].boxes:
            if box.conf > self.config["detection_threshold"] and int(box.cls) in [2, 5, 7]:
                return True
        return False

    def check_receipt(self, frame):
        if not self.config["printer_roi"] or len(self.config["printer_roi"]) != 2:
            return False

        (x1, y1), (x2, y2) = self.config["printer_roi"]
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.config["white_threshold"], 255, cv2.THRESH_BINARY)
        white_area = cv2.countNonZero(thresh)
        return white_area > self.config["min_receipt_area"]

    def get_roi_mask(self, frame, roi_points):
        if len(roi_points) != 2:
            return frame

        line_vec = np.array(roi_points[1]) - np.array(roi_points[0])
        perpendicular = np.array([-line_vec[1], line_vec[0]])
        unit_perp = perpendicular / np.linalg.norm(perpendicular)
        offset = 10
        pts = [
            roi_points[0] + offset * unit_perp,
            roi_points[0] - offset * unit_perp,
            roi_points[1] - offset * unit_perp,
            roi_points[1] + offset * unit_perp
        ]

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    # ===== LOGGING =====
    def log_event(self, frame, is_error=False):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_type = "ERROR" if is_error else "EVENT"
        snapshot_path = os.path.join(self.config['log_path'], "snapshots", f"{timestamp}_{log_type}.jpg")
        log_file = os.path.join(self.config['log_path'], f"{log_type.lower()}_log.txt")

        cv2.imwrite(snapshot_path, frame)
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} - {log_type}: {self.current_status}\n")

        print(f"[{log_type}] Logged at {timestamp}")

    # ===== MAIN LOOP =====
    def run(self):
        if not (self.config["lane_roi"] and len(self.config["lane_roi"]) == 2 and self.config["printer_roi"] and len(self.config["printer_roi"]) == 2):
            ret, frame = self.cap.read()
            if ret:
                self.draw_rois(frame)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame read error - reconnecting...")
                time.sleep(2)
                self.cap = self.init_camera()
                continue

            if not self.vehicle_detected and self.detect_vehicle(frame):
                self.vehicle_detected = True
                self.last_detection_time = time.time()
                self.current_status = "Vehicle detected - waiting for receipt"
                print(self.current_status)

            if self.vehicle_detected:
                elapsed = time.time() - self.last_detection_time

                if elapsed % self.config["check_interval"] < 0.1:
                    if self.check_receipt(frame):
                        self.vehicle_detected = False
                        self.current_status = "Receipt confirmed"
                        print(self.current_status)
                        self.log_event(frame, is_error=False)

                if elapsed >= 4:
                    if not self.check_receipt(frame):
                        self.current_status = "ERROR: No receipt detected"
                        print(self.current_status)
                        self.log_event(frame, is_error=True)
                    self.vehicle_detected = False

            self.display_status(frame)
            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def display_status(self, frame):
        display = frame.copy()
        if self.config["lane_roi"] and len(self.config["lane_roi"]) == 2:
            cv2.line(display, *self.config["lane_roi"], (0, 0, 255), 2)
        if self.config["printer_roi"] and len(self.config["printer_roi"]) == 2:
            cv2.rectangle(display, *self.config["printer_roi"], (0, 255, 0), 2)
        cv2.putText(display, self.current_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Toll Booth Monitor", display)

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    monitor = TollMonitor()
    try:
        monitor.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if monitor.cap.isOpened():
            monitor.cap.release()
        cv2.destroyAllWindows()


Refinements done bro. Code is now cleaned, organized, and error-handling improved. Let me know if you want to add features like real-time dashboard, email alerts, or receipt image saving.

