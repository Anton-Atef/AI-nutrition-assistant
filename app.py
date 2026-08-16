# =====================================================================================
#  🥗 LIVE AI NUTRITION & SMART SHOPPING ASSISTANT
#  Full Streamlit application
#  Model: custom RGB-D ConvNeXt (Nutrition5k) hosted on Hugging Face Hub
#  OCR + Chatbot: Hugging Face Inference API (hosted, no heavy local weights)
# =====================================================================================

import os
import io
import json
import base64
import re
from datetime import datetime, date

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import timm

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download, InferenceClient

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

try:
    from streamlit_lottie import st_lottie
    import requests
    HAS_LOTTIE = True
except Exception:
    HAS_LOTTIE = False

try:
    from gtts import gTTS
    HAS_TTS = True
except Exception:
    HAS_TTS = False


# =====================================================================================
# 1. CONFIG
# =====================================================================================
class CFG:
    MODEL_REPO_ID  = st.secrets.get("MODEL_REPO_ID", "Anton-Atef/AI-nutrition-assistant")
    MODEL_FILENAME = st.secrets.get("MODEL_FILENAME", "best_nutrition_rgbd.pt")

    CHAT_MODEL = st.secrets.get("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    OCR_MODEL  = st.secrets.get("OCR_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    ASR_MODEL  = st.secrets.get("ASR_MODEL", "openai/whisper-large-v3")

    HF_TOKEN = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", None))

    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_COLS_DEFAULT = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

st.set_page_config(
    page_title="AI Nutrition & Shopping Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================================
# 2. THEME / CSS / ANIMATIONS
# =====================================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(160deg, #eafff0 0%, #ffffff 45%, #f2fff6 100%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #1b5e20 0%, #43a047 55%, #9ccc65 100%);
        padding: 34px 40px;
        border-radius: 24px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(27,94,32,0.25);
        animation: fadeInDown 0.7s ease;
    }
    .hero h1 { margin: 0; font-weight: 800; font-size: 2.1rem;}
    .hero p { margin: 6px 0 0 0; opacity: 0.92; font-size: 1.02rem;}

    /* Cards */
    .card {
        background: white;
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        transition: transform .25s ease, box-shadow .25s ease;
        animation: fadeInUp .6s ease;
        margin-bottom: 16px;
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(0,0,0,0.10); }

    .metric-badge {
        display:inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight:600; color:white;
        background: linear-gradient(90deg,#43a047,#2e7d32);
    }

    .macro-pill {
        border-radius: 16px; padding: 14px 16px; color:white; font-weight:600;
        text-align:center; animation: popIn .5s ease;
    }

    div.stButton > button {
        border-radius: 14px !important;
        border: none !important;
        background: linear-gradient(90deg,#2e7d32,#66bb6a) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        transition: all .2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 18px rgba(46,125,50,0.35);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#1b5e20,#2e7d32);
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    @keyframes fadeInDown { from{opacity:0; transform:translateY(-16px);} to{opacity:1; transform:translateY(0);} }
    @keyframes fadeInUp   { from{opacity:0; transform:translateY(16px);}  to{opacity:1; transform:translateY(0);} }
    @keyframes popIn      { from{opacity:0; transform:scale(.9);} to{opacity:1; transform:scale(1);} }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background:#a5d6a7; border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)


def load_lottie(url):
    if not HAS_LOTTIE:
        return None
    try:
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


# =====================================================================================
# 3. MODEL DEFINITION + LOADING  (RGB-D ConvNeXt, matches training notebook)
# =====================================================================================
class NutritionNet(nn.Module):
    def __init__(self, backbone_name, in_chans=4, out_dim=5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


@st.cache_resource(show_spinner=False)
def load_nutrition_model():
    """Downloads best_nutrition_rgbd.pt from the HF Hub and builds the model."""
    ckpt_path = hf_hub_download(
        repo_id=CFG.MODEL_REPO_ID,
        filename=CFG.MODEL_FILENAME,
        token=CFG.HF_TOKEN,
    )
    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE, weights_only=False)

    backbone = ckpt.get("backbone", "convnext_small")
    in_chans = ckpt.get("in_chans", 4)
    target_cols = ckpt.get("target_cols", CFG.TARGET_COLS_DEFAULT)

    model = NutritionNet(backbone, in_chans=in_chans, out_dim=len(target_cols))
    model.load_state_dict(ckpt["model"])
    model.to(CFG.DEVICE)
    model.eval()

    y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.array(ckpt["y_std"], dtype=np.float32)

    return model, y_mean, y_std, target_cols


def estimate_pseudo_depth(rgb_np):
    """No physical depth sensor on a webcam -> heuristic pseudo-depth map.
    (Reduces accuracy vs. true RGB-D but keeps the pipeline fully functional.)"""
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    depth = 1.0 - (blur / 255.0)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth.astype(np.float32)


def preprocess_for_model(pil_img, img_size=CFG.IMG_SIZE):
    rgb = np.array(pil_img.convert("RGB"))
    depth = estimate_pseudo_depth(rgb)

    h, w = rgb.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    rgb_r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = img_size - nh, img_size - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    rgb_p = cv2.copyMakeBorder(rgb_r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    depth_p = cv2.copyMakeBorder(depth_r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    rgb_n = rgb_p.astype(np.float32) / 255.0
    rgb_n = (rgb_n - IMAGENET_MEAN) / IMAGENET_STD
    depth_n = (depth_p[..., None].astype(np.float32) - 0.5) / 0.25

    img4 = np.concatenate([rgb_n, depth_n], axis=2).astype(np.float32)
    x = torch.from_numpy(img4).permute(2, 0, 1).unsqueeze(0)
    return x


@torch.no_grad()
def predict_nutrition(pil_img):
    model, y_mean, y_std, target_cols = load_nutrition_model()
    x = preprocess_for_model(pil_img).to(CFG.DEVICE)
    pred = model(x).cpu().numpy()[0]
    real = pred * y_std + y_mean
    result = {col: max(0.0, float(v)) for col, v in zip(target_cols, real)}
    return result


# =====================================================================================
# 4. HF INFERENCE CLIENT  (Chatbot + OCR + Speech-to-Text)
# =====================================================================================
SYSTEM_PROMPT = """You are an AI Nutrition Coach inside a live nutrition-tracking app.
Help the user understand calories/macros, plan meals, and reach their goal
(weight loss / muscle gain / maintenance). Use ONLY the nutrition data given to you
in the "Trusted data" message — never invent numbers. Be concise, friendly, practical,
never shame the user for any food choice, and never claim to be a doctor."""


@st.cache_resource(show_spinner=False)
def get_hf_client(model_id):
    return InferenceClient(model=model_id, token=CFG.HF_TOKEN)


def rule_based_fallback(user_msg, context):
    msg = user_msg.lower()
    remaining = context.get("remaining", {})
    consumed = context.get("consumed", {})
    if "calor" in msg and ("left" in msg or "remain" in msg):
        return f"You have **{remaining.get('calories',0):.0f} kcal** left for today."
    if "protein" in msg:
        return f"Protein remaining: **{remaining.get('protein',0):.0f} g**."
    if "carb" in msg:
        return f"Carbs remaining: **{remaining.get('carbs',0):.0f} g**."
    if "fat" in msg:
        return f"Fat remaining: **{remaining.get('fat',0):.0f} g**."
    if "eat" in msg and ("what" in msg or "suggest" in msg):
        return "Check the 💡 Meal Suggestions tab — it's built from your exact remaining macros!"
    return ("I'm running in offline mode right now (no HF_TOKEN detected / API busy). "
            f"Quick snapshot — Consumed: {consumed.get('calories',0):.0f} kcal, "
            f"Remaining: {remaining.get('calories',0):.0f} kcal.")


def chat_with_ai(user_msg, context):
    if not CFG.HF_TOKEN:
        return rule_based_fallback(user_msg, context)
    try:
        client = get_hf_client(CFG.CHAT_MODEL)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Trusted data (JSON): {json.dumps(context)}"},
        ]
        for m in st.session_state.chat_history[-6:]:
            messages.append(m)
        messages.append({"role": "user", "content": user_msg})

        resp = client.chat_completion(messages=messages, max_tokens=400, temperature=0.7)
        return resp.choices[0].message.content
    except Exception as e:
        return rule_based_fallback(user_msg, context) + f"\n\n_(API error: {e})_"


OCR_PROMPT = """Read this Nutrition Facts label and return STRICT JSON only, keys:
product_name, serving_size, servings_per_container, calories, total_fat_g,
saturated_fat_g, trans_fat_g, cholesterol_mg, sodium_mg, total_carbohydrates_g,
dietary_fiber_g, total_sugars_g, added_sugars_g, protein_g.
Use null for unreadable values. No markdown, JSON only."""

EXPECTED_KEYS = ["product_name", "serving_size", "servings_per_container", "calories",
                  "total_fat_g", "saturated_fat_g", "trans_fat_g", "cholesterol_mg",
                  "sodium_mg", "total_carbohydrates_g", "dietary_fiber_g",
                  "total_sugars_g", "added_sugars_g", "protein_g"]


def parse_json_from_text(text):
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    json_str = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(json_str)
    except Exception:
        return None


def extract_nutrition_ocr(pil_img):
    if not CFG.HF_TOKEN:
        return None, "No HF_TOKEN set — add it in Streamlit secrets to enable AI OCR."
    try:
        client = get_hf_client(CFG.OCR_MODEL)
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }]
        resp = client.chat_completion(messages=messages, max_tokens=512, temperature=0.1)
        raw = resp.choices[0].message.content
        data = parse_json_from_text(raw)
        if data is None:
            return None, "Could not parse a valid JSON label from the model output."
        clean = {k: data.get(k) for k in EXPECTED_KEYS}
        return clean, None
    except Exception as e:
        return None, f"OCR API error: {e}"


def transcribe_audio(audio_bytes):
    if not CFG.HF_TOKEN:
        return None
    try:
        client = get_hf_client(CFG.ASR_MODEL)
        out = client.automatic_speech_recognition(audio_bytes)
        return out.get("text") if isinstance(out, dict) else str(out)
    except Exception:
        return None


def text_to_speech(text):
    if not HAS_TTS:
        return None
    try:
        tts = gTTS(text=text[:500], lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception:
        return None


# =====================================================================================
# 5. NUTRITION TARGETS / SESSION STATE
# =====================================================================================
ACTIVITY_FACTORS = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55,
                     "Active": 1.725, "Very Active": 1.9}

FOOD_DB = [
    {"name": "Grilled Chicken Breast", "calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    {"name": "Brown Rice (cooked)", "calories": 123, "protein": 2.7, "carbs": 25.6, "fat": 1.0},
    {"name": "Salmon Fillet", "calories": 208, "protein": 20, "carbs": 0, "fat": 13},
    {"name": "Avocado", "calories": 160, "protein": 2, "carbs": 9, "fat": 15},
    {"name": "Greek Yogurt (plain)", "calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4},
    {"name": "Oatmeal (cooked)", "calories": 68, "protein": 2.4, "carbs": 12, "fat": 1.4},
    {"name": "Broccoli", "calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
    {"name": "Sweet Potato", "calories": 86, "protein": 1.6, "carbs": 20, "fat": 0.1},
    {"name": "Eggs (whole)", "calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    {"name": "Almonds", "calories": 579, "protein": 21, "carbs": 22, "fat": 50},
    {"name": "Banana", "calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
    {"name": "Whole Wheat Bread", "calories": 247, "protein": 13, "carbs": 41, "fat": 3.4},
    {"name": "Peanut Butter", "calories": 588, "protein": 25, "carbs": 20, "fat": 50},
    {"name": "Cottage Cheese", "calories": 98, "protein": 11, "carbs": 3.4, "fat": 4.3},
    {"name": "Tofu", "calories": 76, "protein": 8, "carbs": 1.9, "fat": 4.8},
    {"name": "Pizza (slice)", "calories": 266, "protein": 11, "carbs": 33, "fat": 10},
    {"name": "Cheeseburger", "calories": 295, "protein": 17, "carbs": 30, "fat": 14},
    {"name": "Chocolate Cake (slice)", "calories": 371, "protein": 5, "carbs": 51, "fat": 17},
    {"name": "Apple", "calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
    {"name": "Quinoa (cooked)", "calories": 120, "protein": 4.4, "carbs": 21, "fat": 1.9},
]


def init_state():
    ss = st.session_state
    ss.setdefault("profile", {"age": 28, "weight": 75.0, "height": 175.0,
                               "sex": "Male", "activity": "Moderate", "goal": "Maintenance"})
    ss.setdefault("targets", None)
    ss.setdefault("meals", [])          # list of dicts
    ss.setdefault("shopping_list", [])  # list of dicts {item, checked}
    ss.setdefault("chat_history", [])   # list of {role, content}
    ss.setdefault("last_scan", None)
    ss.setdefault("last_ocr", None)


def calculate_targets(profile):
    w, h, age, sex = profile["weight"], profile["height"], profile["age"], profile["sex"]
    bmr = 10 * w + 6.25 * h - 5 * age + (5 if sex == "Male" else -161)
    tdee = bmr * ACTIVITY_FACTORS[profile["activity"]]

    goal = profile["goal"]
    if goal == "Weight Loss":
        calories = tdee - 500
        protein_g = w * 2.2
    elif goal == "Muscle Gain":
        calories = tdee + 400
        protein_g = w * 2.0
    else:
        calories = tdee
        protein_g = w * 1.8

    calories = max(calories, 1200)
    protein_cal = protein_g * 4
    fat_cal = calories * 0.25
    fat_g = fat_cal / 9
    carb_g = max(calories - protein_cal - fat_cal, 0) / 4

    return {"calories": round(calories), "protein": round(protein_g),
            "fat": round(fat_g), "carbs": round(carb_g)}


def today_meals():
    today = date.today().isoformat()
    return [m for m in st.session_state.meals if m["date"] == today]


def consumed_today():
    meals = today_meals()
    return {
        "calories": sum(m["calories"] for m in meals),
        "protein": sum(m["protein"] for m in meals),
        "carbs": sum(m["carbs"] for m in meals),
        "fat": sum(m["fat"] for m in meals),
    }


def remaining_today():
    t = st.session_state.targets or calculate_targets(st.session_state.profile)
    c = consumed_today()
    return {k: round(t[k] - c[k], 1) for k in t}


def add_meal(name, calories, protein, carbs, fat, mass=None, source="manual"):
    st.session_state.meals.append({
        "date": date.today().isoformat(),
        "time": datetime.now().strftime("%H:%M"),
        "name": name,
        "calories": round(calories, 1),
        "protein": round(protein, 1),
        "carbs": round(carbs, 1),
        "fat": round(fat, 1),
        "mass": round(mass, 1) if mass else None,
        "source": source,
    })


# =====================================================================================
# 6. UI PAGES
# =====================================================================================
def render_hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def macro_pill(label, value, unit, color):
    st.markdown(f"""
    <div class="macro-pill" style="background:{color};">
        <div style="font-size:0.85rem; opacity:0.9;">{label}</div>
        <div style="font-size:1.5rem; font-weight:800;">{value}{unit}</div>
    </div>
    """, unsafe_allow_html=True)


def page_dashboard():
    render_hero("Let's Check Your Nutrition Today 🌿",
                "Live overview of calories, macros, and today's meals.")

    if st.session_state.targets is None:
        st.session_state.targets = calculate_targets(st.session_state.profile)

    targets = st.session_state.targets
    consumed = consumed_today()
    remaining = remaining_today()

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=consumed["calories"],
            number={"suffix": " kcal"},
            title={"text": "Calories Consumed Today"},
            gauge={
                "axis": {"range": [0, max(targets["calories"] * 1.3, 500)]},
                "bar": {"color": "#2e7d32"},
                "steps": [
                    {"range": [0, targets["calories"]], "color": "#e8f5e9"},
                    {"range": [targets["calories"], targets["calories"] * 1.3], "color": "#ffe0e0"},
                ],
                "threshold": {"line": {"color": "red", "width": 4},
                              "thickness": 0.8, "value": targets["calories"]},
            },
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Remaining Today")
        c1, c2 = st.columns(2)
        with c1:
            macro_pill("Calories left", f"{remaining['calories']:.0f}", " kcal", "#2e7d32")
            st.write("")
            macro_pill("Protein left", f"{remaining['protein']:.0f}", " g", "#ef6c00")
        with c2:
            macro_pill("Carbs left", f"{remaining['carbs']:.0f}", " g", "#1565c0")
            st.write("")
            macro_pill("Fat left", f"{remaining['fat']:.0f}", " g", "#c2185b")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Macro Progress")
    for label, key, color in [("Protein", "protein", "#ef6c00"),
                               ("Carbs", "carbs", "#1565c0"),
                               ("Fat", "fat", "#c2185b")]:
        frac = min(consumed[key] / targets[key], 1.0) if targets[key] else 0
        st.write(f"**{label}** — {consumed[key]:.0f} / {targets[key]:.0f} g")
        st.progress(frac)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🍽️ Today's Meals")
    meals = today_meals()
    if meals:
        st.dataframe(pd.DataFrame(meals)[["time", "name", "calories", "protein", "carbs", "fat", "source"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No meals logged yet today — scan a food or a label to get started!")
    st.markdown('</div>', unsafe_allow_html=True)


def page_live_scanner():
    render_hero("📸 Live Food Scanner", "Point your camera at a meal to estimate calories & macros.")

    lottie = load_lottie("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        img_file = st.camera_input("Take a photo of your food")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if img_file is not None:
            pil_img = Image.open(img_file)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if lottie and HAS_LOTTIE:
                with st.spinner(""):
                    st_lottie(lottie, height=120, key="scan_anim")
            with st.spinner("🔍 Analyzing food with AI model..."):
                try:
                    result = predict_nutrition(pil_img)
                    st.session_state.last_scan = result
                except Exception as e:
                    st.error(f"Model inference failed: {e}")
                    result = None

            if result:
                st.success("Analysis complete!")
                st.metric("Estimated mass (g)", f"{result['total_mass']:.0f}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Calories", f"{result['total_calories']:.0f} kcal")
                c2.metric("Protein", f"{result['total_protein']:.1f} g")
                c3.metric("Carbs", f"{result['total_carb']:.1f} g")
                c4.metric("Fat", f"{result['total_fat']:.1f} g")

                food_name = st.text_input("Meal name", value="Scanned Meal")
                if st.button("➕ Add to Meal Log", key="add_scan"):
                    add_meal(food_name, result["total_calories"], result["total_protein"],
                              result["total_carb"], result["total_fat"],
                              mass=result["total_mass"], source="camera_ai")
                    st.success(f"Added '{food_name}' to today's log!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Waiting for a photo... 📷")


def page_label_scanner():
    render_hero("🧾 Nutrition Label Scanner", "Scan a packaged product's Nutrition Facts table.")

    tab1, tab2 = st.tabs(["📷 Use Camera", "📁 Upload Image"])
    img = None
    with tab1:
        cam = st.camera_input("Capture the nutrition label", key="label_cam")
        if cam:
            img = Image.open(cam)
    with tab2:
        up = st.file_uploader("Upload a label photo", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up)

    if img is not None:
        st.image(img, caption="Captured Label", width=320)
        if st.button("🔍 Extract Nutrition Data"):
            with st.spinner("Reading label with AI Vision..."):
                data, err = extract_nutrition_ocr(img)
            if err:
                st.error(err)
            else:
                st.session_state.last_ocr = data
                st.success("Label parsed successfully!")

    if st.session_state.last_ocr:
        data = st.session_state.last_ocr
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df = pd.DataFrame(list(data.items()), columns=["Nutrient", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        name = st.text_input("Product name to log", value=data.get("product_name") or "Packaged Food")
        if st.button("➕ Add to Meal Log", key="add_ocr"):
            add_meal(
                name,
                float(data.get("calories") or 0),
                float(data.get("protein_g") or 0),
                float(data.get("total_carbohydrates_g") or 0),
                float(data.get("total_fat_g") or 0),
                source="label_ocr",
            )
            st.success(f"Added '{name}' to today's log!")
        st.markdown('</div>', unsafe_allow_html=True)


def page_meal_log():
    render_hero("📒 Meal Log", "Review, manage, and export everything you've eaten.")

    meals = st.session_state.meals
    if not meals:
        st.info("No meals logged yet.")
        return

    df = pd.DataFrame(meals)
    dates = sorted(df["date"].unique(), reverse=True)
    sel_date = st.selectbox("Filter by date", dates, index=0)
    view = df[df["date"] == sel_date].reset_index(drop=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(view, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Calories", f"{view['calories'].sum():.0f}")
    c2.metric("Total Protein", f"{view['protein'].sum():.0f} g")
    c3.metric("Total Carbs", f"{view['carbs'].sum():.0f} g")
    c4.metric("Total Fat", f"{view['fat'].sum():.0f} g")

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Full Log (CSV)", csv, "meal_log.csv", "text/csv")

    del_idx = st.number_input("Row index to delete (from full log)", min_value=0,
                               max_value=max(len(df) - 1, 0), step=1)
    if st.button("🗑️ Delete Row"):
        st.session_state.meals.pop(del_idx)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def page_goals():
    render_hero("🎯 Goals & Daily Targets", "Tell us about yourself to personalize your targets.")

    p = st.session_state.profile
    with st.form("profile_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 10, 100, p["age"])
        weight = c2.number_input("Weight (kg)", 30.0, 250.0, p["weight"])
        height = c3.number_input("Height (cm)", 100.0, 230.0, p["height"])

        c4, c5, c6 = st.columns(3)
        sex = c4.selectbox("Sex", ["Male", "Female"], index=["Male", "Female"].index(p["sex"]))
        activity = c5.selectbox("Activity Level", list(ACTIVITY_FACTORS.keys()),
                                 index=list(ACTIVITY_FACTORS.keys()).index(p["activity"]))
        goal = c6.selectbox("Goal", ["Weight Loss", "Maintenance", "Muscle Gain"],
                             index=["Weight Loss", "Maintenance", "Muscle Gain"].index(p["goal"]))

        submitted = st.form_submit_button("💾 Save & Recalculate")

    if submitted:
        st.session_state.profile = {"age": age, "weight": weight, "height": height,
                                     "sex": sex, "activity": activity, "goal": goal}
        st.session_state.targets = calculate_targets(st.session_state.profile)
        st.success("Targets updated!")

    targets = st.session_state.targets or calculate_targets(p)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Your Daily Targets")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", f"{targets['calories']} kcal")
    c2.metric("Protein", f"{targets['protein']} g")
    c3.metric("Carbs", f"{targets['carbs']} g")
    c4.metric("Fat", f"{targets['fat']} g")
    st.markdown('</div>', unsafe_allow_html=True)


def page_suggestions():
    render_hero("💡 Smart Meal Suggestions", "Based on your remaining calories & macros right now.")

    remaining = remaining_today()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f"Remaining today → **{remaining['calories']:.0f} kcal**, "
             f"**{remaining['protein']:.0f}g protein**, "
             f"**{remaining['carbs']:.0f}g carbs**, **{remaining['fat']:.0f}g fat**")
    st.markdown('</div>', unsafe_allow_html=True)

    if remaining["calories"] <= 0:
        st.warning("You've hit your calorie target for today — great job! 🎉")
        return

    suggestions = []
    for f in FOOD_DB:
        if f["calories"] <= 0:
            continue
        portion = min((remaining["calories"] / f["calories"]) * 100, 250)
        if portion < 20:
            continue
        scaled = {k: (round(v * portion / 100, 1) if k != "name" else v) for k, v in f.items()}
        scaled["portion_g"] = round(portion)
        suggestions.append(scaled)
    suggestions.sort(key=lambda x: -x["protein"])

    cols = st.columns(3)
    for i, s in enumerate(suggestions[:9]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <h4>🍲 {s['name']}</h4>
                <span class="metric-badge">{s['portion_g']} g portion</span>
                <p style="margin-top:10px;">
                🔥 {s['calories']} kcal &nbsp;|&nbsp; 🥩 {s['protein']}g P<br>
                🍞 {s['carbs']}g C &nbsp;|&nbsp; 🥑 {s['fat']}g F
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Add {s['name']}", key=f"sugg_{i}"):
                add_meal(s["name"], s["calories"], s["protein"], s["carbs"], s["fat"],
                          mass=s["portion_g"], source="suggestion")
                st.success(f"Added {s['name']}!")


def page_shopping():
    render_hero("🛒 Shopping List & Food Compare", "Plan groceries and compare products side by side.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🛍️ Shopping List")
        new_item = st.text_input("Add item")
        if st.button("Add Item") and new_item:
            st.session_state.shopping_list.append({"item": new_item, "checked": False})

        for i, it in enumerate(st.session_state.shopping_list):
            checked = st.checkbox(it["item"], value=it["checked"], key=f"shop_{i}")
            st.session_state.shopping_list[i]["checked"] = checked

        if st.button("🧹 Clear Checked Items"):
            st.session_state.shopping_list = [i for i in st.session_state.shopping_list if not i["checked"]]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("⚖️ Compare Two Foods")
        names = [f["name"] for f in FOOD_DB]
        food_a = st.selectbox("Food A", names, index=0)
        food_b = st.selectbox("Food B", names, index=1)

        fa = next(f for f in FOOD_DB if f["name"] == food_a)
        fb = next(f for f in FOOD_DB if f["name"] == food_b)

        metrics = ["calories", "protein", "carbs", "fat"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name=food_a, x=metrics, y=[fa[m] for m in metrics], marker_color="#43a047"))
        fig.add_trace(go.Bar(name=food_b, x=metrics, y=[fb[m] for m in metrics], marker_color="#ef6c00"))
        fig.update_layout(barmode="group", height=350, title="Per 100g Comparison")
        st.plotly_chart(fig, use_container_width=True)

        better_protein = food_a if fa["protein"] > fb["protein"] else food_b
        better_cal = food_a if fa["calories"] < fb["calories"] else food_b
        st.info(f"💪 Higher protein: **{better_protein}** | 🔥 Lower calories: **{better_cal}**")
        st.markdown('</div>', unsafe_allow_html=True)


def page_chatbot():
    render_hero("🤖 AI Nutrition Chatbot", "Ask about your meals, macros, or get advice — by text or voice!")

    context = {
        "targets": st.session_state.targets or calculate_targets(st.session_state.profile),
        "consumed": consumed_today(),
        "remaining": remaining_today(),
        "goal": st.session_state.profile["goal"],
    }

    if not CFG.HF_TOKEN:
        st.warning("⚠️ No HF_TOKEN found in secrets — chatbot running in limited offline mode.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    colA, colB = st.columns([4, 1])
    with colB:
        speak_reply = st.checkbox("🔊 Voice reply", value=False)
        audio = st.audio_input("🎤 Or speak")

    user_msg = st.chat_input("Ask me anything about your nutrition...")

    if audio is not None and CFG.HF_TOKEN:
        with st.spinner("Transcribing..."):
            text = transcribe_audio(audio.getvalue())
        if text:
            user_msg = text
            st.info(f"🎙️ You said: {text}")

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chat_with_ai(user_msg, context)
            st.markdown(reply)
            if speak_reply:
                audio_buf = text_to_speech(reply)
                if audio_buf:
                    st.audio(audio_buf, format="audio/mp3")

        st.session_state.chat_history.append({"role": "assistant", "content": reply})


# =====================================================================================
# 7. APP ENTRY POINT
# =====================================================================================
def main():
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown("## 🥗 Nutrition AI")
        st.caption("Live camera · OCR · Chatbot · Shopping")
        st.markdown("---")
        st.write(f"**Model:** `{CFG.MODEL_REPO_ID}`")
        st.write(f"**Device:** `{CFG.DEVICE}`")
        st.write("**HF Token:** " + ("✅ Connected" if CFG.HF_TOKEN else "❌ Missing"))
        st.markdown("---")
        st.caption("Built with Streamlit, PyTorch, timm & Hugging Face 🤗")

    pages = {
        "🏠 Dashboard": page_dashboard,
        "📸 Live Scanner": page_live_scanner,
        "🧾 Label Scanner": page_label_scanner,
        "📒 Meal Log": page_meal_log,
        "🎯 Goals": page_goals,
        "💡 Suggestions": page_suggestions,
        "🛒 Shopping & Compare": page_shopping,
        "🤖 AI Chatbot": page_chatbot,
    }

    if HAS_OPTION_MENU:
        selected = option_menu(
            None, list(pages.keys()),
            icons=["house", "camera", "upc-scan", "journal-text",
                   "bullseye", "lightbulb", "cart", "chat-dots"],
            orientation="horizontal",
            styles={
                "container": {"padding": "6px", "background-color": "#f1f8f2", "border-radius": "16px"},
                "nav-link-selected": {"background-color": "#2e7d32"},
            },
        )
    else:
        selected = st.selectbox("Navigate", list(pages.keys()))

    pages[selected]()


if __name__ == "__main__":
    main()
