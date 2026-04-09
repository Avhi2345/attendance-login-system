✨ Key Highlights
🔐 Accurate Face Recognition using DeepFace embeddings
⚡ FastAPI Backend for real-time performance
🧠 Integrated Anti-Spoofing Layer
🎥 Liveness Detection for real-user verification
📊 Automated Attendance Logging System
☁️ Lightweight & Scalable Design
attendance_system/
│── app/
│   ├── main.py
│   ├── antispoof.py
│   ├── liveness.py
│   ├── face_recognition.py
│── database/
│── requirements.txt
│── README.md
**TO RUN THIS PROJECT**
pip install -r requirements.txt
uvicorn app.main:app --reload
