# 🎵 MoodAware — Personalized Emotion-Based Music & Mental Wellness Recommender

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=flat&logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-green?style=flat&logo=mongodb)
![License](https://img.shields.io/badge/License-Academic-purple?style=flat)

> **"Your mood. Your music. Your wellness."**

MoodAware is an AI-powered web application that detects user emotions in real time through facial expression recognition and text-based sentiment analysis, then delivers personalized music recommendations and mental wellness guidance to support emotional well-being.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## 📖 About the Project

Many individuals struggle to manage their emotions due to stress, anxiety, and limited self-awareness. While platforms like Spotify and Apple Music offer mood-based playlists, they rely on listening history rather than detecting a user's **real-time emotional state**.

**MoodAware** addresses this gap by:
- Automatically detecting emotions using **facial expressions** and **text input**
- Providing **personalized music recommendations** that match the current mood
- Offering a **mood tracking dashboard** to help users understand their emotional patterns over time

This project was developed as a **Final Year Undergraduate Thesis** for BSc (Hons) Computer Science at **SLIIT City University / University of Bedfordshire** (Academic Year 2025–2026), under the supervision of **Mr. Alfred Edwin**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 Face Emotion Detection | Real-time facial expression recognition via webcam using CNN/EfficientNetB0 |
| 📝 Text Emotion Analysis | Sentiment classification from user text input using TF-IDF + Logistic Regression |
| 🎵 Music Recommendations | Personalized Spotify & YouTube playlists mapped to detected emotions |
| 🔄 Recovery Pathway | Guided mood progression from negative to positive emotional states |
| 📊 Mood History Dashboard | Visual charts tracking emotional patterns over time |
| 👤 User Authentication | Secure registration and login with Flask-Login and Bcrypt |
| 🛠️ Admin Dashboard | System control hub with user management and emotion analytics |

---

## 🏗️ System Architecture

The system is built on a **three-tier architecture**:

```
┌─────────────────────────────────────────────────────┐
│              CLIENT SIDE (Frontend)                  │
│         HTML  |  CSS  |  JavaScript                  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│           BACKEND SERVER (Python / Flask)            │
│  User Module | Auth Module | Emotion Controller      │
│  Recommendation Engine | Recovery Pathway Module     │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
┌──────────▼──────────┐   ┌──────────▼──────────────┐
│   MongoDB Database   │   │   AI Processing Layer    │
│  Users | Mood History│   │  Face Model (CNN)         │
│  Sessions | Records  │   │  Text Model (TF-IDF/LR)  │
└─────────────────────┘   └─────────────────────────┘
```

### Emotion Detection Flow
1. **User Input** → Face (webcam) or Text
2. **Preprocessing** → OpenCV face detection / NLP tokenization
3. **Emotion Detection** → CNN model / Logistic Regression model
4. **Result Fusion** → Weighted score combination
5. **Recommendation Engine** → Maps mood to music categories
6. **Output** → Detected mood + Spotify/YouTube recommendations + Mood saved to DB

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10**
- **Flask** — Web framework
- **TensorFlow / Keras** — Deep learning model (face emotion)
- **Scikit-learn** — Text emotion classification
- **OpenCV** — Image processing and face detection
- **PyMongo** — MongoDB database integration
- **Flask-Login** — User session management
- **Bcrypt** — Password hashing

### Frontend
- **HTML5 / CSS3 / JavaScript**
- **Bootstrap** — Responsive UI

### Database
- **MongoDB Atlas** — Cloud database for user data and mood history

### AI Models
- **EfficientNetB0 (Transfer Learning)** — Facial emotion recognition
- **TF-IDF + Logistic Regression** — Text-based sentiment analysis

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (or local MongoDB)
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YathavanJD/MoodAware.git
cd MoodAware

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file and add:
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/mood_aware
SECRET_KEY=your_secret_key_here

# 5. Run the application
python app.py
```

### Visit the app
Open your browser and go to: `http://localhost:5000`

---

## 🚀 Usage

1. **Sign Up** — Create an account with a valid email and strong password
2. **Login** — Access your personal dashboard
3. **Detect Mood** — Choose between:
   - **VibeScribe** — Type how you feel and click *Detect Mood*
   - **EmotionCam** — Allow camera access and click *Scan Emotion*
4. **View Results** — See your detected emotion, confidence score, and recommended playlists
5. **Track Mood** — Visit your mood history page to view emotional trends over time

---

## 📈 Model Performance

### Text Emotion Model (TF-IDF + Logistic Regression)
| Metric | Score |
|---|---|
| Overall Accuracy | **87.25%** |
| Macro Avg Precision | 0.92 |
| Macro Avg Recall | 0.85 |
| Macro Avg F1-Score | 0.88 |

**Emotion classes:** joy, sadness, anger, fear, love, surprise, happy, neutral, sad, disgust, angry

### Face Emotion Model (EfficientNetB0 CNN)
| Metric | Score |
|---|---|
| Overall Accuracy | **56%** |
| Macro Avg Precision | 0.53 |
| Macro Avg Recall | 0.48 |
| Macro Avg F1-Score | 0.47 |

**Emotion classes:** angry, disgusted, fearful, happy, neutral, sad, surprised

> Note: Face model accuracy is affected by lighting conditions and image quality. Future improvements include larger, more diverse datasets.

---

## 📸 Screenshots

| Login Page | Dashboard |
|---|---|
| Clean, dark-themed login UI | Dual-input emotion detection interface |

| Emotion Detection Output | Mood History |
|---|---|
| Detected mood with confidence + playlists | Bar chart of emotional patterns over time |

| Admin Dashboard | |
|---|---|
| System Control Hub with user & emotion stats | |

---

## 📁 Project Structure

```
MoodAware/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
│
├── models/
│   ├── face_emotion_model.h5   # Trained CNN face model
│   ├── text_model.pkl          # Trained text emotion model
│   └── vectorizer.pkl          # TF-IDF vectorizer
│
├── training/
│   ├── train_face_model.py     # Face model training script
│   └── train_text_model.py     # Text model training script
│
├── dataset/
│   ├── face_emo/               # FER-2013 face image dataset
│   └── emotion_dataset.csv     # Text emotion dataset
│
├── static/
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript files
│   └── uploads/                # User uploaded images
│
├── templates/
│   ├── index.html              # Landing/Login page
│   ├── dashboard.html          # Main user dashboard
│   ├── profile.html            # User profile page
│   ├── mood_history.html       # Mood history page
│   └── admin/                  # Admin dashboard templates
│
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Expand emotion categories (stress, anxiety, confusion, excitement)
- [ ] Develop a **mobile application** version (iOS/Android)
- [ ] Integrate **voice-based emotion recognition**
- [ ] Improve face model accuracy with larger datasets
- [ ] Add **dark mode** and enhanced UI themes
- [ ] Cloud deployment (AWS / Heroku / Render)
- [ ] Integration with wearable devices for physiological data
- [ ] Enhanced privacy protection for facial recognition processing

---

## 👨‍💻 Author

**Loganathan Yathavan**
- 📧 loganathanyathavan@gmail.com
- 🌐 [Portfolio](https://yathavanjd.github.io/Yathavan_Portfolio/)
- 💼 [LinkedIn](https://www.linkedin.com/in/yathavanloganathan03)
- 🐙 [GitHub](https://github.com/YathavanJD)

---

## 🙏 Acknowledgements

- **Mr. Alfred Edwin** — Project Supervisor, Inivos Consulting Pvt Ltd
- **Mrs. Gayana Fernando** — Lecturer, Undergraduate Project & Research Methodology
- **Mr. Yasas Jayaweera** — Dean, SLIIT City University
- **University of Bedfordshire** & **SLIIT City University** — Academic support and resources
- FER-2013 Dataset, AffectNet, GoEmotions — Training data sources

---

> ⭐ If you found this project useful, please consider giving it a star!

