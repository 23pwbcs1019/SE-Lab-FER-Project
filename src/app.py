import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from collections import deque
from PIL import Image

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="FER System Pro", page_icon="🧠", layout="wide")

# This is the "Dark Tech" CSS you liked
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        border-right: 1px solid #333;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        text-align: center;
        margin-bottom: 10px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #00CCFF; /* The Dark Blue/Cyan Color */
    }
    div.stProgress > div > div > div > div {
        background-color: #00CCFF; /* Progress Bars are Blue */
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'snapshots')
if not os.path.exists(SNAPSHOT_DIR): os.makedirs(SNAPSHOT_DIR)

# --- LOAD LABELS ---
def get_labels():
    data_path = os.path.join(BASE_DIR, 'data', 'raw')
    if os.path.exists(data_path):
        return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and not d.startswith('.')])
    return ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

EMOTIONS = get_labels()

# --- RESOURCES ---
@st.cache_resource
def load_resources():
    model_path = os.path.join(BASE_DIR, 'models', 'emotion_model.h5')
    if not os.path.exists(model_path): return None, None
    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

# --- PREDICT ---
def predict_emotion(face_img, model):
    roi = cv2.resize(face_img, (48, 48))
    # Note: Removed "Histogram Equalization" to fix the Shadow=Happy bug
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0).reshape(1, 48, 48, 1)
    return model.predict(roi, verbose=0)[0]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    
    app_mode = st.radio("System Mode", ["🔴 Live Analysis", "📂 Image Upload"])
    
    st.markdown("### 🎚️ Calibration")
    confidence_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.caption(f"Model: CNN-V3 | Classes: {len(EMOTIONS)}")
    
    if app_mode == "🔴 Live Analysis":
        run_detection = st.checkbox("Activate Camera", value=False)
        enable_snapshot = st.button("📸 Capture Snapshot")
        if 'last_snap' in st.session_state:
            st.success(f"Saved: {st.session_state.last_snap}")

# --- MAIN APP ---
model, face_cascade = load_resources()

if model is None:
    st.error("❌ Model missing. Please run 'src/train_model.py' to restore accuracy.")
    st.stop()

col1, col2 = st.columns([2, 1])

# ================= LIVE MODE =================
if app_mode == "🔴 Live Analysis":
    st_frame = col1.empty()
    st_metrics = col2.empty()
    prediction_history = deque(maxlen=5)

    if run_detection:
        cap = cv2.VideoCapture(0)
        while run_detection:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30,30))
            
            current_pred = "Scanning..."
            final_probs = np.zeros(len(EMOTIONS))
            display_color = (100, 100, 100)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                raw_preds = predict_emotion(face_roi, model)
                
                prediction_history.append(raw_preds)
                avg_preds = np.mean(prediction_history, axis=0)
                max_index = np.argmax(avg_preds)
                max_prob = avg_preds[max_index]
                
                # SLIDER LOGIC
                if max_prob > confidence_threshold:
                    if max_index < len(EMOTIONS):
                        current_pred = EMOTIONS[max_index]
                        final_probs = avg_preds
                        # Blue/Green for positive, Red for negative
                        display_color = (0, 255, 0) if current_pred in ['Happy', 'Surprise'] else (0, 0, 255)
                else:
                    current_pred = "Neutral / Low Confidence"
                    display_color = (255, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), display_color, 2)
                cv2.putText(frame, f"{current_pred} {int(max_prob*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

            if enable_snapshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snap_{timestamp}.jpg"
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), frame)
                st.session_state.last_snap = filename
                enable_snapshot = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", width="stretch")
            
            with st_metrics.container():
                st.markdown(f"""
                <div class="metric-card">
                    <p style="color:#888; margin:0;">Detected Emotion</p>
                    <p class="big-font">{current_pred}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Probability Distribution")
                for i, emotion in enumerate(EMOTIONS):
                    if i < len(final_probs):
                        st.progress(float(final_probs[i]), text=f"{emotion}")
        cap.release()

# ================= UPLOAD MODE =================
elif app_mode == "📂 Image Upload":
    st.markdown("### 📂 Static Image Analysis")
    
    # === UPDATED LINE: Added 'tiff' and 'tif' support ===
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'tiff', 'tif'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) == 0:
            st.warning("⚠️ No face detected. Analyzing full image...")
            faces = [[0, 0, gray.shape[1], gray.shape[0]]]
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            preds = predict_emotion(face_roi, model)
            label = EMOTIONS[np.argmax(preds)]
            
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="big-font">{label}</p>
                </div>
                """, unsafe_allow_html=True)
                for i, emo in enumerate(EMOTIONS):
                    st.progress(float(preds[i]), text=f"{emo}")
        
        col1.image(img_array, channels="RGB", width="stretch")