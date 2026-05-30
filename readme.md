# Object Detection System using YOLOv4

## 📌 Project Overview

This project is a real-time Object Detection System built using **Python**, **OpenCV**, and **YOLOv4**.

The system uses your webcam to detect objects in real time and displays:

* Object labels
* Confidence scores
* Bounding boxes around detected objects

YOLOv4 is a powerful deep learning model trained on the COCO dataset for detecting multiple objects efficiently.

---

# 🚀 Features

* Real-time object detection
* Webcam integration
* YOLOv4 model support
* Bounding boxes with labels
* Confidence percentage display
* Fast and accurate detection

---

# 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* YOLOv4
* COCO Dataset

---

# 📂 Project Structure

```plaintext
ODS/
│
├── app.py
│
├── yolo-coco-data/
│   ├── coco.names
│   ├── yolov4.cfg
│   └── yolov4.weights
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/object-detection-system.git
```

## Step 2: Open the Project Folder

```bash
cd object-detection-system
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python app.py
```

The webcam will open and start detecting objects automatically.

Press **Q** to close the application.

---

# 📦 Required Files

Make sure these files are present inside the `yolo-coco-data` folder:

* `coco.names`
* `yolov4.cfg`
* `yolov4.weights`

---

# 📥 Download YOLOv4 Weights

Download the YOLOv4 weights file from the official source:

https://pjreddie.com/media/files/yolov4.weights

After downloading, place the file inside:

```plaintext
yolo-coco-data/
```

---

# 📸 Sample Output

The system detects objects and displays:

* Object name
* Detection confidence
* Green bounding box

Example:

```plaintext
person 98.45%
chair 87.32%
bottle 91.10%
```

# 🎯 Future Improvements

* Image upload support
* Video file detection
* Voice alerts
* Object counting
* Flask web interface
* Mobile deployment

## GitHub Repository

Repository Link:

https://github.com/manvithaindukuru/object-detection-system
