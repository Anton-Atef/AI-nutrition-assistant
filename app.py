# =========================================================
#  🥗 AI Nutrition & Smart Shopping Assistant — Streamlit
#  - Nutrition5k RGB-D regression model (weights from HF Hub)
#  - HF Inference API for:
#      Chat: Qwen/Qwen2.5-7B-Instruct
#      OCR : Qwen/Qwen2-VL-7B-Instruct (Vision)
#      ASR : openai/whisper-large-v3
# =========================================================

import os, io, re, json, time, base64
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import streamlit as st

import torch
import torch.nn as nn
import timm

from huggingface_hub import hf_hub_download
from huggingface_hub import InferenceClient


# -----------------------------
# ✅ PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Nutrition & Smart Shopping Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# 🎨 GLOBAL CSS / ANIMATIONS
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

/* Animated gradient header */
.main-header {
    background: linear-gradient(-45deg, #00C853, #76FF03, #1B5E20, #00E676);
    background-size: 400% 400%;
    animation: gradientShift 9s ease infinite;
    padding: 26px 18px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.main-header h1 { margin: 0; font-size: 2.0rem; font-weight: 700; letter-spacing: 0.2px; }
.main-header p  { margin: 6px 0 0; opacity: 0.95; }

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.86);
    border-radius: 16px;
    padding: 16px 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    border: 1px solid rgba(0,200,83,0.18);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 30px rgba(0,200,83,0.22);
}

/* Metric pill */
.metric-pill {
    display:inline-block;
    padding:6px 12px;
    border-radius: 999px;
    background: linear-gradient(90deg,#00C853,#1B5E20);
    color: white;
    font-weight: 600;
    font-size: 0.82rem;
    margin-left: 8px;
    animation: pop 0.45s ease;
}
@keyframes pop { from{transform:scale(0.75); opacity:0;} to{transform:scale(1); opacity:1;} }

/* Progress bar */
.progress-wrap {
    background: #E8F5E9;
    border-radius: 14px;
    height: 18px;
    overflow: hidden;
    margin: 8px 0 14px;
    border: 1px solid rgba(27,94,32,0.10);
}
.progress-fill {
    height: 100%;
    border-radius: 14px;
    background: linear-gradient(90deg,#76FF03,#00C853,#1B5E20);
    animation: growBar 1.2s ease-out;
}
@keyframes growBar { from { width: 0%; } }

/* Chat bubbles */
.chat-user {
    background:#DCF8C6;
    padding:10px 14px;
    border-radius:14px 14px 0 14px;
    margin:7px 0;
    display:inline-block;
    max-width:82%;
    float:right;
    clear:both;
    box-shadow: 0 8px 18px rgba(0,0,0,0.05);
}
.chat-bot  {
    background:#F1F0F0;
    padding:10px 14px;
    border-radius:14px 14px 14px 0;
    margin:7px 0;
    display:inline-block;
    max-width:82%;
    float:left;
    clear:both;
    box-shadow: 0 8px 18px rgba(0,0,0,0.05);
}

/* Live indicator */
.pulse-dot {
    height:12px; width:12px; background:#FF1744; border-radius:50%;
    display:inline-block; margin-right:8px;
    animation: pulse 1.3s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,23,68,0.60); }
    70% { box-shadow: 0 0 0 12px rgba(255,23,68,0.00); }
    100% { box-shadow: 0 0 0 0 rgba(255,23,68,0.00); }
}

footer { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
  <h1>🥗 Live AI Nutrition & Smart Shopping Assistant</h1>
  <p>Food photo → calories & macros • Nutrition label OCR • Meal log • Smart suggestions • AI Coach • Voice</p>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# ⚙️ CONFIG
# -----------------------------
class CFG:
    MODEL_REPO_ID = "Anton-Atef/AI-nutrition-assistant"
    MODEL_FILENAME = "best_nutrition_rgbd.pt"

    BACKBONE = "convnext_small"
    IN_CHANS = 4
    IMG_SIZE = 256
    TARGET_COLS = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]

    CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    OCR_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
    ASR_MODEL = "openai/whisper-large-v3"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Mild speed tuning on CPU
try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
except Exception:
    pass


def get_hf_token() -> Optional[str]:
    """Read token from Streamlit secrets first, then environment."""
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = None
    return token or os.environ.get("HF_TOKEN")


# -----------------------------
# 🧠 MODEL DEFINITION
# -----------------------------
class NutritionNet(nn.Module):
    def __init__(self, backbone_name: str, in_chans: int = 4, out_dim: int = 5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=in_chans,
            num_classes=0,
            global_pool="avg",
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


@st.cache_resource(show_spinner="🔽 Downloading Nutrition model from Hugging Face (first run only)...")
def load_model():
    token = get_hf_token()
    ckpt_path = hf_hub_download(
        repo_id=CFG.MODEL_REPO_ID,
        filename=CFG.MODEL_FILENAME,
        token=token,  # works for private repos too
    )

    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)

    backbone = ckpt.get("backbone", CFG.BACKBONE)
    in_chans = ckpt.get("in_chans", CFG.IN_CHANS)
    target_cols = ckpt.get("target_cols", CFG.TARGET_COLS)

    model = NutritionNet(backbone, in_chans=in_chans, out_dim=len(target_cols))
    model.load_state_dict(ckpt["model"])
    model.eval().to(CFG.DEVICE)

    y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
    y_std = np.array(ckpt["y_std"], dtype=np.float32)

    return model, y_mean, y_std, target_cols


# -----------------------------
# 🖼️ PREPROCESS + PREDICT
# -----------------------------
def preprocess_image(pil_img: Image.Image, img_size: int = CFG.IMG_SIZE) -> torch.Tensor:
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]

    scale = img_size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    rgb_r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    top, left = (img_size - nh) // 2, (img_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = rgb_r

    rgb_norm = canvas.astype(np.float32) / 255.0
    rgb_norm = (rgb_norm - IMAGENET_MEAN) / IMAGENET_STD

    # No real depth sensor in typical webcam/phone.
    # Use neutral pseudo-depth centered at 0 after normalization.
    depth = np.full((img_size, img_size, 1), 0.5, dtype=np.float32)
    depth = (depth - 0.5) / 0.25

    img4 = np.concatenate([rgb_norm, depth], axis=2).astype(np.float32)
    x = torch.from_numpy(img4).permute(2, 0, 1).unsqueeze(0)  # (1,4,H,W)
    return x


def predict_nutrition(pil_img: Image.Image) -> Dict[str, float]:
    model, y_mean, y_std, target_cols = load_model()
    x = preprocess_image(pil_img).to(CFG.DEVICE)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    pred = pred * y_std + y_mean
    pred = np.clip(pred, 0, None)
    return {k: float(v) for k, v in zip(target_cols, pred.tolist())}


# -----------------------------
# 🤖 HF INFERENCE CLIENTS (cached)
# -----------------------------
@st.cache_resource
def hf_client_text():
    token = get_hf_token()
    return InferenceClient(model=CFG.CHAT_MODEL, token=token) if token else None


@st.cache_resource
def hf_client_ocr():
    token = get_hf_token()
    return InferenceClient(model=CFG.OCR_MODEL, token=token) if token else None


@st.cache_resource
def hf_client_asr():
    token = get_hf_token()
    return InferenceClient(model=CFG.ASR_MODEL, token=token) if token else None


# -----------------------------
# 🧾 OCR via Qwen2-VL (Vision)
# -----------------------------
OCR_JSON_PROMPT = """
Carefully read the provided Nutrition Facts label and extract the nutritional information into a strict JSON object.

Use EXACTLY these keys:
- "product_name" (string or null)
- "serving_size" (string or null)
- "servings_per_container" (number only or null)
- "calories" (number only or null)
- "total_fat_g" (number only or null)
- "saturated_fat_g" (number only or null)
- "trans_fat_g" (number only or null)
- "cholesterol_mg" (number only or null)
- "sodium_mg" (number only or null)
- "total_carbohydrates_g" (number only or null)
- "dietary_fiber_g" (number only or null)
- "total_sugars_g" (number only or null)
- "added_sugars_g" (number only or null)
- "protein_g" (number only or null)

Rules:
1) Extract ONLY numeric values for nutrients (exclude units like g/mg).
2) Do NOT confuse % Daily Value with grams/mg.
3) If not visible or not reliable, set it to null. NEVER guess.
4) Return ONLY valid JSON (no markdown, no commentary).
""".strip()


def _data_url_from_pil(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_json_object(text: str) -> Optional[dict]:
    # Find first {...} block, best-effort
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def extract_nutrition_ocr_hf(pil_img: Image.Image) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    client = hf_client_ocr()
    if client is None:
        return None, "HF_TOKEN missing. Add HF_TOKEN in Streamlit Secrets to enable OCR."

    try:
        image_url = _data_url_from_pil(pil_img)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": OCR_JSON_PROMPT},
                ],
            }
        ]

        resp = client.chat_completion(
            messages=messages,
            max_tokens=450,
            temperature=0.1,
        )
        out_text = resp.choices[0].message["content"]
        parsed = _extract_json_object(out_text)
        if not isinstance(parsed, dict):
            return None, "OCR model did not return valid JSON. Try a clearer label photo."

        parsed["raw_model_text"] = out_text
        return parsed, None

    except Exception as e:
        return None, f"OCR failed: {e}"


# -----------------------------
# 🎙️ ASR via Whisper-large-v3 (HF)
# -----------------------------
def transcribe_audio_hf(audio_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    client = hf_client_asr()
    if client is None:
        return None, "HF_TOKEN missing. Add HF_TOKEN in Streamlit Secrets to enable voice."

    try:
        # Try common InferenceClient task names across versions
        if hasattr(client, "automatic_speech_recognition"):
            res = client.automatic_speech_recognition(audio=audio_bytes)
            # Can be str or dict-like depending on backend
            if isinstance(res, str):
                return res, None
            if isinstance(res, dict) and "text" in res:
                return res["text"], None

        if hasattr(client, "audio_to_text"):
            res = client.audio_to_text(audio_bytes)
            if isinstance(res, str):
                return res, None
            if isinstance(res, dict) and "text" in res:
                return res["text"], None

        return None, "ASR method not available in this huggingface_hub version."
    except Exception as e:
        return None, f"ASR failed: {e}"


# -----------------------------
# 🧭 SIMPLE INTENT ROUTER
# -----------------------------
class Route:
    def __init__(self, intent: str, confidence: float):
        self.intent = intent
        self.confidence = confidence


class AIRouter:
    def __init__(self):
        self.rules = {
            "daily_summary": [
                r"how many calories.*(left|remaining)",
                r"calories.*(left|remaining)",
                r"what did i eat today",
                r"daily.*summary",
                r"today.*nutrition",
            ],
            "remaining_macros": [
                r"remaining.*(protein|carb|fat|macro)",
                r"how much protein.*(left|remaining)",
                r"how much.*carb.*(left|remaining)",
                r"how much.*fat.*(left|remaining)",
            ],
            "user_targets": [
                r"my.*target",
                r"my macros",
                r"calorie target",
                r"protein target",
            ],
            "recommendation": [
                r"what should i eat",
                r"recommend.*meal",
                r"what.*eat.*(dinner|lunch|breakfast)",
            ],
            "comparison": [
                r"compare",
                r"which.*better",
                r"better.*between",
            ],
        }

    def route(self, message: str) -> Route:
        text = re.sub(r"\s+", " ", message.lower().strip())
        scores = {k: sum(bool(re.search(p, text)) for p in pats) for k, pats in self.rules.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Route("general", 0.25)
        return Route(best, min(0.95, 0.55 + scores[best] * 0.15))


ROUTER = AIRouter()

SYSTEM_PROMPT = (
    "You are an AI Nutrition Coach inside a nutrition tracking app.\n"
    "Use ONLY the trusted user/app data provided to you in the context.\n"
    "Never invent calories/macros.\n"
    "Be concise, supportive, and never shame the user.\n"
    "Do not claim to be a doctor; for medical questions advise consulting a professional.\n"
)


# -----------------------------
# 🗂️ SESSION STATE
# -----------------------------
defaults = {
    "meal_log": [],
    "shopping_list": [],
    "chat_history": [],
    "goal": "Maintain Weight",
    "targets": {"calories": 2000, "protein": 130, "carb": 230, "fat": 65},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def totals_today() -> Dict[str, float]:
    if not st.session_state.meal_log:
        return {"calories": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
    df = pd.DataFrame(st.session_state.meal_log)
    return {
        "calories": float(df["total_calories"].sum()),
        "protein": float(df["total_protein"].sum()),
        "carb": float(df["total_carb"].sum()),
        "fat": float(df["total_fat"].sum()),
    }


def add_to_log(name: str, nutrition: Dict[str, Any]):
    st.session_state.meal_log.append(
        {
            "time": time.strftime("%H:%M:%S"),
            "food": name,
            "total_mass": float(nutrition.get("total_mass", 0) or 0),
            "total_calories": float(nutrition.get("total_calories", nutrition.get("calories", 0)) or 0),
            "total_fat": float(nutrition.get("total_fat", nutrition.get("total_fat_g", 0)) or 0),
            "total_carb": float(nutrition.get("total_carb", nutrition.get("total_carbohydrates_g", 0)) or 0),
            "total_protein": float(nutrition.get("total_protein", nutrition.get("protein_g", 0)) or 0),
        }
    )


# -----------------------------
# 🎯 SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### 🎯 Goal & Daily Targets")

    st.session_state.goal = st.selectbox(
        "Goal",
        ["Lose Weight", "Maintain Weight", "Gain Muscle"],
        index=["Lose Weight", "Maintain Weight", "Gain Muscle"].index(st.session_state.goal),
    )

    with st.expander("🧮 Auto-calculate targets (Mifflin-St Jeor)"):
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        age = st.number_input("Age", 10, 90, 25)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        height = st.number_input("Height (cm)", 120, 220, 170)
        activity = st.select_slider(
            "Activity level",
            options=[1.2, 1.375, 1.55, 1.725, 1.9],
            value=1.55,
            format_func=lambda x: {
                1.2: "Sedentary",
                1.375: "Light",
                1.55: "Moderate",
                1.725: "Active",
                1.9: "Very Active",
            }[x],
        )

        if st.button("⚡ Calculate Targets", use_container_width=True):
            bmr = 10 * weight + 6.25 * height - 5 * age + (5 if sex == "Male" else -161)
            tdee = bmr * activity
            if st.session_state.goal == "Lose Weight":
                tdee *= 0.8
            elif st.session_state.goal == "Gain Muscle":
                tdee *= 1.15

            protein = weight * (2.0 if st.session_state.goal == "Gain Muscle" else 1.6)
            fat = (tdee * 0.25) / 9
            carb = (tdee - protein * 4 - fat * 9) / 4

            st.session_state.targets = {
                "calories": int(round(tdee)),
                "protein": int(round(protein)),
                "carb": int(round(max(carb, 0))),
                "fat": int(round(fat)),
            }
            st.success("Targets updated!")

    st.markdown("#### Manual Override")
    t = st.session_state.targets
    t["calories"] = st.number_input("Calories target", 500, 6000, int(t["calories"]))
    t["protein"] = st.number_input("Protein target (g)", 10, 400, int(t["protein"]))
    t["carb"] = st.number_input("Carbs target (g)", 10, 600, int(t["carb"]))
    t["fat"] = st.number_input("Fat target (g)", 10, 300, int(t["fat"]))

    st.divider()
    st.caption("🔐 Add `HF_TOKEN` in Streamlit Secrets to enable Chat + OCR + Voice (and private model access).")


# -----------------------------
# 🧭 TABS
# -----------------------------
tab_scan, tab_ocr, tab_dash, tab_shop, tab_chat = st.tabs(
    ["📷 Live Scan", "🧾 Label OCR", "📊 Dashboard", "🛒 Shopping & Compare", "🤖 AI Coach + Voice"]
)


# =========================================================
# 📷 TAB 1 — LIVE SCAN
# =========================================================
with tab_scan:
    st.markdown('<span class="pulse-dot"></span> **Camera Scan — point at your food**', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        cam_img = st.camera_input("Capture your meal photo", key="scan_cam")
        upl_img = st.file_uploader("…or upload a food image", type=["jpg", "jpeg", "png"], key="scan_upl")
        food_name = st.text_input("Food name (optional)", placeholder="e.g., Chicken Rice Bowl")

    image_to_use = None
    if cam_img is not None:
        image_to_use = Image.open(cam_img)
    elif upl_img is not None:
        image_to_use = Image.open(upl_img)

    with c2:
        if image_to_use is not None:
            st.image(image_to_use, caption="Input image", use_container_width=True)

            if st.button("🔍 Analyze Nutrition", type="primary", use_container_width=True):
                with st.spinner("Running Nutrition5k model…"):
                    result = predict_nutrition(image_to_use)
                st.session_state["last_scan"] = result
                st.session_state["last_scan_name"] = food_name.strip() or "Scanned Food"

    if "last_scan" in st.session_state:
        r = st.session_state["last_scan"]
        st.markdown("### 🥘 Estimated Nutrition")
        cols = st.columns(5)
        labels = [
            ("total_mass", "Mass (g)"),
            ("total_calories", "Calories"),
            ("total_protein", "Protein (g)"),
            ("total_carb", "Carbs (g)"),
            ("total_fat", "Fat (g)"),
        ]
        for col, (k, lab) in zip(cols, labels):
            col.markdown(
                f"""
<div class="glass-card" style="text-align:center;">
  <div style="font-size:1.55rem;font-weight:700;color:#1B5E20;margin-bottom:2px;">
    {r.get(k, 0.0):.1f}
  </div>
  <div style="opacity:0.85;">{lab}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        if st.button("➕ Add to Meal Log", use_container_width=True):
            add_to_log(st.session_state.get("last_scan_name", "Scanned Food"), r)
            st.success("Added to meal log ✅")
            try:
                st.balloons()
            except Exception:
                pass


# =========================================================
# 🧾 TAB 2 — NUTRITION LABEL OCR (Qwen2-VL)
# =========================================================
with tab_ocr:
    st.markdown("Upload or capture a packaged product label (**Nutrition Facts**).")

    a, b = st.columns(2)

    with a:
        label_cam = st.camera_input("Capture label", key="ocr_cam")
        label_upl = st.file_uploader("…or upload label image", type=["jpg", "jpeg", "png"], key="ocr_upl")

    label_img = None
    if label_cam is not None:
        label_img = Image.open(label_cam)
    elif label_upl is not None:
        label_img = Image.open(label_upl)

    with b:
        if label_img is not None:
            st.image(label_img, caption="Label image", use_container_width=True)

            if st.button("📖 Extract Nutrition (AI OCR)", type="primary", use_container_width=True):
                with st.spinner("Reading label with AI OCR…"):
                    data, err = extract_nutrition_ocr_hf(label_img)

                if err:
                    st.error(err)
                else:
                    st.session_state["last_ocr"] = data
                    st.success("Extracted ✅")

    if "last_ocr" in st.session_state:
        data = st.session_state["last_ocr"].copy()
        raw_model_text = data.pop("raw_model_text", None)

        st.markdown("### ✅ Extracted Nutrition JSON")
        st.code(json.dumps(data, indent=2), language="json")

        df_show = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
        st.dataframe(df_show, use_container_width=True)

        prod_name = st.text_input("Product name for log", value=data.get("product_name") or "Packaged Food")
        if st.button("➕ Add Label Item to Meal Log", use_container_width=True):
            add_to_log(prod_name, data)
            st.success("Added to meal log ✅")

        with st.expander("Debug: raw model output"):
            st.write(raw_model_text or "")


# =========================================================
# 📊 TAB 3 — DASHBOARD
# =========================================================
with tab_dash:
    st.markdown("### 📆 Today's Summary")

    tot = totals_today()
    tgt = st.session_state.targets

    def bar(label: str, consumed: float, target: float, unit: str = ""):
        pct = 0 if target <= 0 else min(100.0, (consumed / target) * 100.0)
        remaining = max(target - consumed, 0.0)

        st.markdown(
            f"**{label}** — {consumed:.0f}/{target:.0f}{unit} "
            f"<span class='metric-pill'>remaining {remaining:.0f}{unit}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='progress-wrap'><div class='progress-fill' style='width:{pct:.1f}%;'></div></div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        bar("🔥 Calories", tot["calories"], float(tgt["calories"]), unit=" kcal")
        bar("💪 Protein", tot["protein"], float(tgt["protein"]), unit=" g")
    with c2:
        bar("🍞 Carbs", tot["carb"], float(tgt["carb"]), unit=" g")
        bar("🥑 Fat", tot["fat"], float(tgt["fat"]), unit=" g")

    st.markdown("### 🍽️ Meal Log")
    if st.session_state.meal_log:
        st.dataframe(pd.DataFrame(st.session_state.meal_log), use_container_width=True)
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            if st.button("🗑️ Clear Meal Log", use_container_width=True):
                st.session_state.meal_log = []
                st.rerun()
        with cc2:
            if st.button("⬇️ Download CSV", use_container_width=True):
                df = pd.DataFrame(st.session_state.meal_log)
                st.download_button(
                    "Download",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="meal_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("No meals logged yet — scan a meal or OCR a label to start.")

    st.markdown("### 💡 Smart Suggestions (based on remaining)")
    remaining_cal = float(tgt["calories"]) - tot["calories"]
    remaining_pro = float(tgt["protein"]) - tot["protein"]

    SUGGESTIONS = {
        "Lose Weight": [
            "Grilled chicken + salad + light dressing",
            "Greek yogurt + berries + chia",
            "Egg omelet (2 eggs) + veggies",
        ],
        "Maintain Weight": [
            "Salmon + rice + veggies",
            "Turkey wrap + fruit",
            "Tofu stir-fry + noodles",
        ],
        "Gain Muscle": [
            "Chicken + rice + veggies bowl",
            "Protein smoothie (milk + banana + whey)",
            "Beef + sweet potato + salad",
        ],
    }

    if remaining_cal <= 0:
        st.warning("You’ve reached your calorie target. If still hungry: go for lighter, protein-rich foods.")
    else:
        for s in SUGGESTIONS[st.session_state.goal]:
            st.markdown(
                f"<div class='glass-card' style='margin-bottom:10px;'>"
                f"🍴 {s}"
                f"<span class='metric-pill'>fits ~{max(0,int(remaining_cal/3))} kcal</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(f"Remaining (approx): {remaining_cal:.0f} kcal, {remaining_pro:.0f} g protein")


# =========================================================
# 🛒 TAB 4 — SHOPPING & COMPARE
# =========================================================
with tab_shop:
    st.markdown("### 🛒 Smart Shopping List")

    new_item = st.text_input("Add an item")
    if st.button("➕ Add Item", use_container_width=True) and new_item.strip():
        st.session_state.shopping_list.append(new_item.strip())

    if st.session_state.shopping_list:
        for i, item in enumerate(list(st.session_state.shopping_list)):
            r1, r2 = st.columns([6, 1])
            r1.markdown(f"<div class='glass-card'>• {item}</div>", unsafe_allow_html=True)
            if r2.button("❌", key=f"del_{i}", use_container_width=True):
                st.session_state.shopping_list.pop(i)
                st.rerun()
    else:
        st.info("Shopping list is empty.")

    st.markdown("---")
    st.markdown("### ⚖️ Compare Two Products (manual)")

    def manual_entry(prefix: str) -> Dict[str, Any]:
        st.markdown(f"**{prefix}**")
        name = st.text_input(f"{prefix} name", key=f"{prefix}_name")
        cal = st.number_input(f"{prefix} calories", 0, 3000, 0, key=f"{prefix}_cal")
        pro = st.number_input(f"{prefix} protein (g)", 0, 200, 0, key=f"{prefix}_pro")
        carb = st.number_input(f"{prefix} carbs (g)", 0, 300, 0, key=f"{prefix}_carb")
        fat = st.number_input(f"{prefix} fat (g)", 0, 200, 0, key=f"{prefix}_fat")
        sugar = st.number_input(f"{prefix} sugar (g)", 0, 200, 0, key=f"{prefix}_sugar")
        return {"name": name or prefix, "calories": cal, "protein": pro, "carb": carb, "fat": fat, "sugar": sugar}

    x, y = st.columns(2)
    with x:
        A = manual_entry("Product A")
    with y:
        B = manual_entry("Product B")

    if st.button("⚖️ Compare", type="primary", use_container_width=True):
        comp = pd.DataFrame([A, B]).set_index("name")
        st.dataframe(comp, use_container_width=True)

        # Simple heuristic: lower calories & sugar wins (can be expanded)
        a_key = (A["calories"], A["sugar"])
        b_key = (B["calories"], B["sugar"])
        winner = A["name"] if a_key <= b_key else B["name"]
        st.success(f"✅ Better pick (lower calories/sugar): **{winner}**")


# =========================================================
# 🤖 TAB 5 — AI COACH + VOICE
# =========================================================
with tab_chat:
    st.markdown("### 🤖 AI Coach (uses your logged meals + targets)")

    def build_context() -> Dict[str, Any]:
        tot = totals_today()
        tgt = st.session_state.targets
        return {
            "goal": st.session_state.goal,
            "daily_target": tgt,
            "consumed_today": tot,
            "remaining": {k: round(float(tgt[k]) - tot[k], 1) for k in ["calories", "protein", "carb", "fat"]},
            "meal_log": st.session_state.meal_log,
        }

    def rule_reply(message: str, intent: str, ctx: Dict[str, Any]) -> str:
        r = ctx["remaining"]
        if intent == "daily_summary":
            c = ctx["consumed_today"]
            return (
                f"Today: **{c['calories']:.0f} kcal**, **{c['protein']:.0f}g protein**, "
                f"**{c['carb']:.0f}g carbs**, **{c['fat']:.0f}g fat**. "
                f"Remaining: **{r['calories']:.0f} kcal**."
            )
        if intent == "remaining_macros":
            return (
                f"Remaining today → 🔥{r['calories']:.0f} kcal, "
                f"💪{r['protein']:.0f}g protein, 🍞{r['carb']:.0f}g carbs, 🥑{r['fat']:.0f}g fat."
            )
        if intent == "user_targets":
            t = ctx["daily_target"]
            return f"Targets ({ctx['goal']}): {t['calories']} kcal • {t['protein']}P • {t['carb']}C • {t['fat']}F (grams)."
        if intent == "recommendation":
            return (
                f"You have about **{r['calories']:.0f} kcal** left. "
                f"To improve protein, aim for a lean protein + veggies + a carb portion that fits."
            )
        return "Ask me about remaining calories/macros, meal suggestions, or comparing foods."

    def llm_reply(message: str, ctx: Dict[str, Any]) -> Optional[str]:
        client = hf_client_text()
        if client is None:
            return None
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + "\nTrusted context:\n" + json.dumps(ctx)},
                {"role": "user", "content": message},
            ]
            resp = client.chat_completion(messages=messages, max_tokens=400, temperature=0.7)
            return resp.choices[0].message["content"]
        except Exception:
            return None

    # Render history
    for m in st.session_state.chat_history:
        cls = "chat-user" if m["role"] == "user" else "chat-bot"
        st.markdown(f"<div class='{cls}'>{m['content']}</div>", unsafe_allow_html=True)
    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

    user_msg = st.chat_input("Ask about your day, remaining macros, meal ideas…")

    with st.expander("🎙️ Voice input (Whisper via HF)"):
        if hasattr(st, "audio_input"):
            audio = st.audio_input("Record your question")
            if audio is not None and st.button("Transcribe & Send", use_container_width=True):
                with st.spinner("Transcribing…"):
                    text, err = transcribe_audio_hf(audio.getvalue())
                if err:
                    st.error(err)
                else:
                    st.info(f"Transcribed: {text}")
                    user_msg = text
        else:
            st.info("Your Streamlit version does not support st.audio_input yet.")

    if user_msg:
        ctx = build_context()
        route = ROUTER.route(user_msg)

        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.spinner("Thinking…"):
            reply = llm_reply(user_msg, ctx)
            if not reply:
                reply = rule_reply(user_msg, route.intent, ctx)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


st.markdown("---")
st.caption("⚠️ General nutrition guidance only — not medical advice.")
