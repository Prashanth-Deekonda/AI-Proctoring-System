# 🎓 AI-Based Automated Exam Proctoring System

An AI-powered automated exam proctoring system developed using **Python, OpenCV, MediaPipe, and YOLOv8** to monitor students during exams and detect cheating activities such as unauthorized persons and mobile phone usage.

---

## 📌 Overview

This system uses **Computer Vision and Artificial Intelligence** to monitor students in real time through a webcam. It automatically verifies student identity and detects suspicious activities.

The system helps reduce human invigilator effort and ensures secure online examinations.

---

## 🚀 Features

- ✅ Face Detection using MediaPipe  
- ✅ Face Registration and Verification  
- ✅ Multiple Person Detection using YOLOv8  
- ✅ Mobile Phone Detection using YOLOv8  
- ✅ Real-time Monitoring  
- ✅ Automatic Cheating Alerts  
- ✅ Bounding box visualization  
- ✅ High accuracy and fast detection  

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- YOLOv8 (Ultralytics)
- NumPy
- Webcam (for real-time input)

---

## 📂 Project Structure

AI-Proctoring-System/
│
├── main.py # Final integrated proctoring system
│
├── Phase1/
│ └── facedetection.py # Face detection module
│
├── Phase2/
│ ├── register.py # Face registration
│ └── verify.py # Face verification
│
├── models/
│ └── object_detector.py # YOLO object detection
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ Installation

### Step 1: Clone the repository

git clone https://github.com/Prashanth-Deekonda/AI-Proctoring-System.git

cd AI-Proctoring-System


---

### Step 2: Create virtual environment

python -m venv venv


Activate:

Windows:
venv\Scripts\activate


---

### Step 3: Install dependencies

pip install -r requirements.txt


---

## ▶️ How to Run

### Step 1: Register Student Face

python Phase2/register.py


Press **S** to register face.

---

### Step 2: Run AI Proctoring System

python main.py


---

## 🧠 System Working

The system performs the following steps:

1. Captures webcam video
2. Detects face using MediaPipe
3. Verifies student identity
4. Detects multiple persons using YOLOv8
5. Detects mobile phone usage
6. Generates alert if cheating detected
7. Displays results in real time

---

## ⚠️ Alerts Generated

- Unauthorized person detected
- Multiple persons detected
- Mobile phone detected
- No face detected

---

## 📊 Applications

- Online examinations
- Remote learning monitoring
- Secure certification exams
- Educational institutions

---

## 🔮 Future Scope

- Eye movement tracking
- Audio monitoring
- Cloud-based monitoring
- Web-based interface
- Integration with online exam portals

---

## 👨‍💻 Author

**Prashanth Deekonda**  
B.Tech Mini Project  
AI-Based Automated Exam Proctoring System  

---

## 📜 License

This project is for educational purposes.
