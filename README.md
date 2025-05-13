# **Toll Booth Monitoring System**  

**A lightweight, CPU-optimized system for automated toll booth monitoring using computer vision.**  

## **📌 Overview**  
This Python-based system detects vehicles entering a toll lane and verifies if a receipt is printed within **4 seconds**. If no receipt is found, it logs the event with a timestamp and snapshot. Designed for **CPU-only environments**, it uses a hybrid detection approach for reliability without GPU acceleration.  

### **✨ Key Features**  
✔ **Vehicle Detection** – Hybrid motion + YOLOv8n verification  
✔ **Receipt Validation** – White pixel counting + shape checks  
✔ **Background Service** – Runs with system tray controls (Start/Pause/Stop)  
✔ **Auto-Recovery** – Handles RTSP disconnections gracefully  
✔ **Lightweight Logging** – Saves errors with snapshots  

## **🛠 Technologies Used**  
- **OpenCV** (Real-time video processing)  
- **YOLOv8n** (Lightweight vehicle verification)  
- **pystray** (System tray GUI)  
- **BackgroundSubtractorMOG2** (Motion detection)  

## **⚡ Optimized for CPU**  
- Processes only **ROI regions** (reduces compute load)  
- Runs at **15-20 FPS** on modern CPUs  
- Adaptive thresholding for receipt detection  

## **📂 Logging & Output**  
- **`/logs/events.log`** – Successful detections  
- **`/logs/errors.log`** – Failed receipt checks  
- **`/logs/snapshots/`** – Archived error frames  

---

### **🚀 Getting Started**  
```bash
pip install ultralytics opencv-python numpy pystray pillow
python toll_service.py  # Runs in system tray
```

**Configurable via `config.json`:**  
```json
{
  "rtsp_url": "rtsp://your_stream",
  "resolution": [1280, 720],
  "vehicle_classes": [2, 5, 7]  # car, bus, truck
}
```

---

**Ideal for:**  
- Toll operators needing **automated receipt verification**  
- Edge device deployments (Raspberry Pi, Intel NUC)  
- Low-power 24/7 monitoring  

**Contributions welcome!** 🛠️  

---  
**License:** MIT | **Author:** [Your Name]
