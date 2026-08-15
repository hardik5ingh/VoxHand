# VoxHand 🖐️🗣️

VoxHand is a real-time American Sign Language (ASL) gesture recognition and translation platform powered by computer vision and machine learning. It captures hand landmarks via webcam, classifies gestures in real-time, converts signs into text and voice, and integrates Google OAuth 2.0 for secure authentication.

---

## 🚀 Live Demo

* **Frontend Application:** [https://voxhand.netlify.app](https://voxhand.netlify.app)
* **Backend API:** [https://voxhand-backend.onrender.com](https://voxhand-backend.onrender.com)

---

## ✨ Key Features

* **Real-Time Landmark Detection:** Utilizes MediaPipe Hand Landmarker (`hand_landmarker.task`) to track 21 3D hand keypoints directly from video frames.
* **Machine Learning Classification:** Scikit-Learn classifier trained to identify ASL letters and custom hand gestures with high accuracy.
* **Voice & Speech Synthesis:** Converts recognized gestures into real-time spoken audio using the Web Speech API.
* **Google OAuth 2.0 Authentication:** End-to-end user authentication verifying Google Identity JWT tokens securely on the backend.
* **Fully Containerized Backend:** Production-grade Docker container configured with Linux graphics dispatch libraries (`libglvnd`, `libgl1`, `libegl1`, `libgles2`).
* **Data Collection & Training Pipeline:** Modular scripts to record custom gesture datasets and retrain the model on demand.

---

## 🛠️ Tech Stack

**Frontend:**
* HTML5, CSS3, Vanilla JavaScript (ES6+)
* Google Identity Services SDK (`accounts.google.com/gsi/client`)
* Deployed on **Netlify**

**Backend:**
* Python 3.10, Flask, Flask-CORS
* MediaPipe Tasks Vision, OpenCV, Scikit-Learn, NumPy
* Production WSGI: Gunicorn
* Containerized & Deployed on **Render**

---

## 📁 Repository Structure

```text
VoxHand/
├── backend/
│   ├── data/                   # Collected dataset CSV files
│   ├── models/                 # Model artifacts
│   │   ├── gesture_model.pkl   # Trained gesture classification model
│   │   └── hand_landmarker.task# MediaPipe vision landmarker model
│   ├── app.py                  # Main Flask API server and endpoints
│   ├── collect_data.py         # Script to capture custom landmark data
│   ├── train_model.py          # Script to train Scikit-Learn model
│   ├── merge_asl_dataset.py    # Merges custom and ASL datasets
│   ├── realtime_translator.py  # Local desktop testing script
│   ├── Dockerfile              # Production Docker configuration
│   └── requirements.txt        # Python backend dependencies
├── frontend/
│   ├── images/                 # Static UI assets and icons
│   ├── index.html              # Frontend UI layout
│   ├── script.js               # Camera stream, prediction calls & auth handling
│   └── style.css               # Styling and responsiveness
└── README.md
```

---

## ⚙️ Local Installation & Setup

### 1. Prerequisites
* Python 3.10+
* Git
* VS Code (with the Live Server extension)
* A working webcam

### 2. Clone the Repository
```bash
git clone [https://github.com/hardik5ingh/VoxHand.git](https://github.com/hardik5ingh/VoxHand.git)
cd VoxHand
```

### 3. Backend Setup
```powershell
cd backend
python -m venv voxhand_env
voxhand_env\Scripts\activate
pip install -r requirements.txt
python app.py
```
*(Server runs locally at `http://localhost:5000`)*

### 4. Frontend Setup
1. In `frontend/script.js`, set `const API_BASE = "http://localhost:5000";`.
2. Right-click `frontend/index.html` in VS Code $\rightarrow$ select **Open with Live Server** (`http://localhost:5500`).

---

## 📖 How to Use Locally

1. **Sign In with Google:** Click the Google button in the header.
2. **Start Camera:** Grant webcam permission to activate the live feed.
3. **Perform Signs:** Place your hand in front of the lens for real-time detection and voice speech playback.
4. **Collect & Retrain (Optional):** Run `python collect_data.py` to record new signs and `python train_model.py` to update the model.

---

## ☁️ Deployment Guide

### Backend (Render Docker Service)
1. Push code to GitHub.
2. Create a new **Web Service** on Render.
3. Select **Docker** environment with Root Directory set to `backend`.
4. Add environment variable `GOOGLE_CLIENT_ID` with your Google Cloud client ID.
5. Deploy the service.

### Frontend (Netlify)
1. Import repository on Netlify.
2. Set **Base directory** to `frontend`.
3. Set **Publish directory** to `frontend`.
4. Deploy the site.

---

## 📡 API Reference

#### `GET /api/config`
Fetches public client configuration.
```json
{
  "google_client_id": "your_client_id.apps.googleusercontent.com"
}
```

#### `POST /api/auth/google`
Authenticates a user via Google OAuth JWT token.
```json
{
  "token": "<GOOGLE_ID_TOKEN>"
}
```

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).