import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from bson.objectid import ObjectId
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from collections import Counter
from werkzeug.utils import secure_filename
import random
import warnings
warnings.filterwarnings("ignore")


# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = "mood_healing_ai_secret"
app.config["MONGO_URI"] = "mongodb://localhost:27017/mood_aware"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mongo = PyMongo(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------- LOAD MODELS ----------------
text_model = pickle.load(open("text_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
face_model = load_model("face_emotion_model.h5", compile=False)

FACE_CLASSES = ["anger","disgust","fear","joy","neutral","sadness","surprise"]

# ---------------- MOODS ----------------
MOODS = {
    "sadness":{
        "emoji":"😢",
        "quote":"Slow healing… brighter days ahead 💙",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DWVrtsSlLKzro",
            "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=k4V3Mo61fJM&list=PLIudw_RzT2eMVybyXNWizMAYHd8Ge3z62",
            "https://www.youtube.com/results?search_query=sad+calm+uplifting+songs+playlis",
            "https://music.youtube.com/playlist?list=RDTMAK5uy_nUxNnI_Sq9nwrocdgHNgOzlPGDBBx6WD4",
            "https://youtube.com/playlist?list=PLJQrUSvEtfTjRsUq6tbDCZ1g7cRP2gy9n&si=7t9ZWj_s-32BPTrP",
            "https://www.youtube.com/watch?v=K3Qzzggn--s&list=PLgzTt0k8mXzHcKebL8d0uYHfawiARhQja",
            "https://www.youtube.com/watch?v=HaMq2nn5ac0&list=PLzeh4p9T2kiA1Kv63Llne68KRiQnRwnZx"        
                            ]
    },
    "anger":{
        "emoji":"😠",
        "quote":"Breathe… release the fire 🔥",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DX4sWSp46o6pS",
            "https://open.spotify.com/playlist/37i9dQZF1DWXLeA8Omikj7"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=CGj85pVzRJs&list=PLIudw_RzT2eMotJBqwHscJtH-tMFW8PQn",
            "https://www.youtube.com/watch?v=8SbUC-UaAxE",
            "https://www.youtube.com/watch?v=7SwhmMpuZjU&list=PLQHAzJTOiMlBqIgKRtoq5j0hMk_qxnhvB",
            "https://www.youtube.com/watch?v=Vrr3lRLjZ1Y&list=PLknqyEOvGo1YgL11BN1m-YOxaFHl29elY"
        ]
    },
    "fear":{
        "emoji":"😨",
        "quote":"You are stronger than fear.",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DX7KNK00Vto9s",
            "https://open.spotify.com/playlist/37i9dQZF1DWZo5GuZsWYwN"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=QUQsqBqxoR4&list=PLIudw_RzT2eOEhl9z9vQZwS26Ilt9gR-3",
            "https://www.youtube.com/watch?v=1vrEljMfXYo",
            "https://www.youtube.com/watch?v=ooOak5FVkpM&list=PLJQrUSvEtfTjRsUq6tbDCZ1g7cRP2gy9n",
            "https://www.youtube.com/watch?v=f8TkUMJtK5k&list=PL9urKWGhmkzpypLVBxFxBAaif57vSe8eG",
            "https://www.youtube.com/watch?v=wXxArQbJ0YY&list=PLJQrUSvEtfThRUURhl0gkS_EnIAc9r308"
        ]
    },
    "joy":{
        "emoji":"😄",
        "quote":"Keep shining ✨",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
            "https://open.spotify.com/playlist/37i9dQZF1DX0BcQWzuB7ZO"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=ru0K8uYEZWw&list=PLIudw_RzT2eMZ05zRxCvhpYXgPmZVQEnK",
            "https://www.youtube.com/watch?v=d-diB65scQU",
            "https://www.youtube.com/watch?v=n4BXrDPuEyU&list=PLcq_bRTniqI6Sy6DfS_Rb8mLjuaMu8KTk",
            "https://www.youtube.com/watch?v=h8uKldEUrPE&list=PLa-vfb_PfSLwuoJqbdNr3Ix5jsgANBsBx",
            "https://youtube.com/playlist?list=PLBpnsidVh94gy_7oVrBdeFuRGGPKu5jTO&si=K72Ppuqc_lZ2Epyk"
        ]
    },
    "love":{
        "emoji":"❤️",
        "quote":"Feel the warmth of love.",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DX7r9qSioS5Lh",
            "https://open.spotify.com/playlist/37i9dQZF1DX5IDTimEWoTd"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g&list=PLIudw_RzT2eNYz0OiB3tD_6Tg0R1nbXvA",
            "https://www.youtube.com/watch?v=n0XoS1NHTs0",
            "https://youtube.com/playlist?list=PL3-sRm8xAzY8LhlTyJ2uf-EcQSm_vzSqw&si=a2naBse4Jsv6WCEz",
            "https://youtube.com/playlist?list=PLgzTt0k8mXzE6H9DDgiY7Pd8pKZteis48&si=ueXodiLi2Wmp5_E9",
            "https://youtube.com/playlist?list=PLx2Jv96o522ORh69HaDKClrz2Midqj8AE&si=dZqsFA8MHhoQlg6q"
        ]
    },
    "surprise":{
        "emoji":"😲",
        "quote":"Embrace the unexpected!",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DX4JpneCYUI78",
            "https://open.spotify.com/playlist/37i9dQZF1DX9qNs32fujYe"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=ZbZSe6N_BXs&list=PLIudw_RzT2eNuH5ir9F_N-kbvb72j-wwz",
            "https://www.youtube.com/watch?v=oRdxUFDoQe0",
            "https://youtube.com/playlist?list=PLyhpbO3GZhVKuczU1zLAwQcb1NNAdj7of&si=hAzykUF7DmJJf8ls",
            "https://youtube.com/playlist?list=PL8Xp9LjKcouW-JeElXtZSu0YU79-Oc77O&si=Em0viXIn-V_CNnUZ",
            "https://youtube.com/playlist?list=PLjIydaclej7C1NDDi_vMmlKy7UeIZ_e9Z&si=LjZYe9-BdAziCYtr"
        ]
    },
    "neutral":{
        "emoji":"😐",
        "quote":"Calm is power.",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU",
            "https://open.spotify.com/playlist/37i9dQZF1DX4sWSp46o6pS"
        ],
        "youtube":[
            "https://music.youtube.com/playlist?list=RDTMAK5uy_nUxNnI_Sq9nwrocdgHNgOzlPGDBBx6WD4",
            "https://youtube.com/playlist?list=PL3fNjt3zNn7_1U2QpBxQ_368rC84jfDTO&si=ZpuWpdJsDuQTSlkq"
        ]
    },
    "disgust":{
        "emoji":"🤢",
        "quote":"Refresh your mind 🍃",
        "spotify":[
            "https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU",
            "https://open.spotify.com/playlist/37i9dQZF1DX5h3CJ4bFrka"
        ],
        "youtube":[
            "https://www.youtube.com/watch?v=lFcSrYw-ARY",
            "https://www.youtube.com/watch?v=JGwWNGJdvx8",
            "https://youtube.com/playlist?list=PL8TV4m3E3io5A2TqCj2k1970YTBRuyWvT&si=HfeBXN50-6eHrkoJ"
        ]
    }
}

# ---------------- RECOVERY PATH ----------------
RECOVERY_PATH = {
    "sadness": ["sadness", "neutral", "joy"],
    "anger": ["anger", "neutral", "joy"],
    "fear": ["fear", "neutral", "joy"],
    "disgust": ["disgust", "neutral"],
    "neutral": ["neutral", "joy"],
    "joy": ["joy"],
    "love": ["love", "joy"],
    "surprise": ["surprise", "joy"]
}
# ---------------- GLOBAL TEMPLATE ----------------
@app.context_processor
def inject_moods():
    return dict(MOODS=MOODS)

# ---------------- USER CLASS ----------------
class User(UserMixin):
    def __init__(self, user):
        self.id = str(user["_id"])
        self.username = user["username"]
        self.role = user.get("role", "user")
        self.profile_image = user.get("profile_image", None)

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(user)
    return None

# ---------------- SAVE MOOD ----------------
def save_mood(user_id, mood, source):
    mongo.db.moods.insert_one({
        "user_id": user_id,
        "mood": mood,
        "source": source,
        "created_at": datetime.utcnow()
    })
# ---------------- RECOVERY PLAYLIST GENERATOR ----------------
def generate_recovery_playlist(mood):
    path = RECOVERY_PATH.get(mood, ["neutral"])
    playlist = []

    for stage in path:
        mood_data = MOODS.get(stage, {})

        spotify_link = random.choice(mood_data["spotify"]) if mood_data.get("spotify") else None
        youtube_link = random.choice(mood_data["youtube"]) if mood_data.get("youtube") else None

        playlist.append({
            "stage": stage,
            "emoji": mood_data.get("emoji"),
            "quote": mood_data.get("quote"),
            "spotify": spotify_link,
            "youtube": youtube_link
        })

    return playlist
# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# ---------------- SIGNUP WITH CONFIRM PASSWORD ----------------
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Check password match
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("signup"))

        # Check if email exists
        if mongo.db.users.find_one({"email": email}):
            flash("Email already exists!", "danger")
            return redirect(url_for("signup"))

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

        file = request.files.get("profile_image")
        filename = None
        if file and file.filename != "":
            filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        mongo.db.users.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password,
            "role": "user",
            "profile_image": filename,
            "created_at": datetime.utcnow()
        })

        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# ---------------- LOGIN ------------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = mongo.db.users.find_one({"email": email})
        if user and bcrypt.check_password_hash(user["password"], password):
            if user.get("is_banned"):
                flash("Your account is banned.", "danger")
                return redirect(url_for("login"))
            login_user(User(user))
            return redirect(url_for("dashboard"))

        flash("Invalid email or password", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- FORGOT PASSWORD ----------------
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')

        user = mongo.db.users.find_one({"username": username, "email": email})
        if not user:
            flash("Username or email is incorrect!", "danger")
            return redirect(url_for('forgot_password'))

        session['reset_user_id'] = str(user['_id'])
        flash("Verification successful! Please set your new password.", "success")
        return redirect(url_for('reset_password_direct'))

    return render_template("forgot_password.html")

@app.route('/reset-password-direct', methods=['GET', 'POST'])
def reset_password_direct():
    if 'reset_user_id' not in session:
        flash("Unauthorized access. Please verify your username and email first.", "danger")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        hashed = bcrypt.generate_password_hash(new_password).decode("utf-8")

        mongo.db.users.update_one(
            {"_id": ObjectId(session['reset_user_id'])},
            {"$set": {"password": hashed}}
        )

        session.pop('reset_user_id', None)

        flash("Password reset successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template("reset_password.html")

# ---------------- DASHBOARD --------------------
@app.route("/dashboard", methods=["GET","POST"])
@login_required
def dashboard():
    result = None

    if request.method == "POST" and "text" in request.form:
        text = request.form["text"]
        vec = vectorizer.transform([text])
        mood = text_model.predict(vec)[0].lower().strip()

        mood = "surprise" if mood == "suprise" else mood
        mood = mood if mood in MOODS else "neutral"

        save_mood(current_user.id, mood, "text")

        # ✅ FIX: Proper indentation
        recovery_playlist = generate_recovery_playlist(mood)

        result = {
            "mood": mood,
            "emoji": MOODS[mood]["emoji"],
            "quote": MOODS[mood]["quote"],
            "recovery_playlist": recovery_playlist
        }

    elif "face_result" in session:
        result = session.pop("face_result")

    return render_template("index.html", result=result)


# ---------------- FACE MOOD --------------------
@app.route("/face", methods=["POST"])
@login_required
def face():
    try:
        img_data = request.json["image"].split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(img_data))).convert("L").resize((48,48))

        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 48, 48, 1)

        mood_idx = np.argmax(face_model.predict(arr)[0])
        mood = FACE_CLASSES[mood_idx].lower()
        mood = mood if mood in MOODS else "neutral"

        save_mood(current_user.id, mood, "face")

        # ✅ FIX: Proper indentation
        recovery_playlist = generate_recovery_playlist(mood)

        session["face_result"] = {
            "mood": mood,
            "emoji": MOODS[mood]["emoji"],
            "quote": MOODS[mood]["quote"],
            "recovery_playlist": recovery_playlist
        }

        return jsonify({"ok": True})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------- PROFILE -------------------------
@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    user = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form.get("password")
        file = request.files.get("profile_image")

        update_data = {"username": username, "email": email}

        if password:
            update_data["password"] = bcrypt.generate_password_hash(password).decode("utf-8")

        if file and file.filename != "":
            filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            update_data["profile_image"] = filename

        mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": update_data})

        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    moods = list(mongo.db.moods.find({"user_id": current_user.id}).sort("created_at",-1))
    for m in moods:
        m["_id"] = str(m["_id"])

    return render_template("profile.html", user=user, moods=moods)


# ---------------- EDIT PROFILE -------------------------
@app.route("/edit_profile", methods=["GET", "POST"])
@login_required
def edit_profile():
    user = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form.get("password")
        file = request.files.get("profile_image")

        update_data = {"username": username, "email": email}

        if password:
            update_data["password"] = bcrypt.generate_password_hash(password).decode("utf-8")

        if file and file.filename != "":
            filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            update_data["profile_image"] = filename

        mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": update_data})

        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    return render_template("edit_profile.html", user=user)


# ---------------- ADMIN ----------------
@app.route("/admin")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        return redirect(url_for("dashboard"))

    users = list(mongo.db.users.find())
    moods = list(mongo.db.moods.find().sort("created_at", -1))

    weekly_chart_data = [12, 18, 9, 15, 20, 14, 22]

    start_of_today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    today_count = mongo.db.moods.count_documents({
        "created_at": {"$gte": start_of_today}
    })

    return render_template(
        "admin_dashboard.html",
        users=users,
        moods=moods,
        weekly_chart_data=weekly_chart_data,
        today_count=today_count
    )


@app.route("/admin/toggle-ban/<user_id>", methods=["POST"])
@login_required
def admin_toggle_ban(user_id):
    if current_user.role != "admin":
        return redirect(url_for("dashboard"))

    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

    if user:
        new_status = not user.get("is_banned", False)
        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_banned": new_status}}
        )

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete-user/<user_id>", methods=["POST"])
@login_required
def admin_delete_user(user_id):
    if current_user.role != "admin":
        return redirect(url_for("dashboard"))

    mongo.db.users.delete_one({"_id": ObjectId(user_id)})
    return redirect(url_for("admin_dashboard"))


# ---------------- USER MOOD STATS ----------------
@app.route("/api/user-mood-stats")
@login_required
def user_mood_stats():
    moods = list(mongo.db.moods.find({"user_id": current_user.id}))

    mood_list = [m["mood"] for m in moods]
    mood_counts = Counter(mood_list)

    for mood in MOODS.keys():
        if mood not in mood_counts:
            mood_counts[mood] = 0

    return jsonify(mood_counts)


if __name__ == "__main__":
    app.run(debug=True)