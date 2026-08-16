# app.py
"""
🥗 AI Nutrition & Smart Shopping Assistant — Combined, polished Streamlit app
Features:
 - Camera / Upload food scan -> Nutrition RGB-D ConvNeXt model (weights from HF)
 - Optional MiDaS monocular depth (best-effort, optional)
 - Nutrition Label OCR (pytesseract offline fallback, HF Vision LLM if HF_TOKEN set)
 - AI Coach: HF Inference chat (Qwen...) if HF_TOKEN set, otherwise fast rule-based fallback
 - Voice: optional speech-to-text and text-to-speech when libs available
 - Dashboard, meal log, shopping list, food compare, targets calculator
"""

import os, io, json, time, re, base64
from datetime import date, datetime
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# Deep learning
import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download

# Optional features
try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_OK = True
except Exception:
    HF_CLIENT_OK = False

try:
    import pytesseract
    TESS_OK = True
except Exception:
    TESS_OK = False

try:
    from gtts import gTTS
    GTTS_OK = True
except Exception:
    GTTS_OK = False

try:
    import speech_recognition as sr
    SR_OK = True
except Exception:
    SR_OK = False

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(page_title="AI Nutrition & Smart Shopping Assistant", page_icon="🥗", layout="wide")

class CFG:
    HF_REPO   = st.secrets.get("MODEL_REPO_ID", "Anton-Atef/AI-nutrition-assistant")
    HF_FILE   = st.secrets.get("MODEL_FILENAME", "best_nutrition_rgbd.pt")
    BACKBONE  = "convnext_small"
    IN_CHANS  = 4
    IMG_SIZE  = 256
    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
    CHAT_MODEL = st.secrets.get("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    OCR_MODEL  = st.secrets.get("OCR_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    ASR_MODEL  = st.secrets.get("ASR_MODEL", "openai/whisper-large-v3")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def get_hf_token():
    # prefer Streamlit secrets, then environment
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        pass
    return token or os.environ.get("HF_TOKEN", None)

HF_TOKEN = get_hf_token()

# ----------------------------
# Styling (clean & friendly)
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main-hero {
  background: linear-gradient(90deg,#66bb6a,#2e7d32);
  color: white; padding: 22px; border-radius: 14px; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(46,125,50,0.12);
}
.card { background: rgba(255,255,255,0.98); padding:14px; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.04); margin-bottom:12px; }
.metric-pill { display:inline-block; padding:6px 12px; border-radius:16px; background:linear-gradient(90deg,#43A047,#2E7D32); color:white; font-weight:600; }
.chat-user { background:#DCF8C6; padding:10px 14px; border-radius:14px 14px 0 14px; margin:6px 0; display:inline-block; max-width:80%; float:right; clear:both;}
.chat-bot  { background:#F1F0F0; padding:10px 14px; border-radius:14px 14px 14px 0; margin:6px 0; display:inline-block; max-width:80%; float:left; clear:both;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-hero"><h2>🥗 AI Nutrition & Smart Shopping Assistant</h2><div>Camera-based food recognition · Label OCR · Macro tracking · AI Coach</div></div>', unsafe_allow_html=True)

# ----------------------------
# Model (matches training)
# ----------------------------
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
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource(show_spinner="🔽 Downloading AI model from Hugging Face Hub (first run only)...")
def load_nutrition_model():
    token = HF_TOKEN
    try:
        ckpt_path = hf_hub_download(repo_id=CFG.HF_REPO, filename=CFG.HF_FILE, token=token)
    except Exception as e:
        st.error(f"Could not download model from HF Hub: {e}")
        raise

    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)
    backbone = ckpt.get("backbone", CFG.BACKBONE)
    in_chans = ckpt.get("in_chans", CFG.IN_CHANS)
    target_cols = ckpt.get("target_cols", ["total_mass","total_calories","total_fat","total_carb","total_protein"])

    model = NutritionNet(backbone, in_chans=in_chans, out_dim=len(target_cols))
    model.load_state_dict(ckpt["model"])
    model.to(CFG.DEVICE)
    model.eval()

    y_mean = np.array(ckpt.get("y_mean", np.zeros(len(target_cols))), dtype=np.float32)
    y_std  = np.array(ckpt.get("y_std", np.ones(len(target_cols))), dtype=np.float32)
    return model, y_mean, y_std, target_cols

# ----------------------------
# Optional MiDaS depth (best-effort)
# ----------------------------
@st.cache_resource(show_spinner=False)
def try_load_midas():
    # This is optional and only used if the user toggles 'Use MiDaS depth' in the UI.
    # It may take time to download and is not required for the app to function.
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(CFG.DEVICE).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        return midas, transform
    except Exception as e:
        # don't fail hard — fallback to pseudo-depth
        return None, None

def compute_midas_depth(midas_tuple, pil_img):
    midas, transform = midas_tuple
    if midas is None:
        return None
    img = np.array(pil_img.convert("RGB"))
    input_batch = transform(Image.fromarray(img)).to(CFG.DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    # normalize to 0..1
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
    return prediction.astype(np.float32)

def estimate_pseudo_depth(rgb_np):
    # simple heuristic: blurred luminance inversion -> depth-like map
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    depth = 1.0 - (blur / 255.0)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth.astype(np.float32)

# ----------------------------
# Preprocessing & prediction
# ----------------------------
def preprocess_image(pil_img, img_size=CFG.IMG_SIZE, use_midas=False, midas_tuple=(None,None)):
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]
    # compute depth channel (MiDaS if available and requested, else pseudo)
    if use_midas and midas_tuple[0] is not None:
        depth_map = compute_midas_depth(midas_tuple, pil_img)
        if depth_map is None:
            depth_map = estimate_pseudo_depth(rgb)
    else:
        depth_map = estimate_pseudo_depth(rgb)

    scale = img_size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    rgb_r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    depth_r = cv2.resize(depth_map, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    depth_canvas = np.full((img_size, img_size), 0.5, dtype=np.float32)

    top, left = (img_size - nh) // 2, (img_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = rgb_r
    depth_canvas[top:top+nh, left:left+nw] = depth_r

    rgb_norm = canvas.astype(np.float32) / 255.0
    rgb_norm = (rgb_norm - IMAGENET_MEAN) / IMAGENET_STD

    depth_ch = ((depth_canvas[..., None].astype(np.float32) - 0.5) / 0.25)  # normalized similar to training
    img4 = np.concatenate([rgb_norm, depth_ch], axis=2).astype(np.float32)
    x = torch.from_numpy(img4).permute(2,0,1).unsqueeze(0)
    return x

def predict_nutrition(pil_img, use_midas=False, midas_tuple=(None,None)):
    model, y_mean, y_std, target_cols = load_nutrition_model()
    x = preprocess_image(pil_img, CFG.IMG_SIZE, use_midas=use_midas, midas_tuple=midas_tuple).to(CFG.DEVICE)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    pred = pred * y_std + y_mean
    pred = np.clip(pred, 0, None)
    return dict(zip(target_cols, pred.tolist()))

# ----------------------------
# OCR: pytesseract (fallback) and HF Vision LLM (if HF_TOKEN)
# ----------------------------
NUM_PAT = r"([\d]+\.?[\d]*)"
OCR_PATTERNS = {
    "calories":              rf"calories\D{{0,12}}{NUM_PAT}",
    "total_fat_g":           rf"total\s*fat\D{{0,12}}{NUM_PAT}\s*g",
    "saturated_fat_g":       rf"saturated\s*fat\D{{0,12}}{NUM_PAT}\s*g",
    "trans_fat_g":           rf"trans\s*fat\D{{0,12}}{NUM_PAT}\s*g",
    "cholesterol_mg":        rf"cholesterol\D{{0,12}}{NUM_PAT}\s*mg",
    "sodium_mg":             rf"sodium\D{{0,12}}{NUM_PAT}\s*mg",
    "total_carbohydrates_g": rf"total\s*carb\w*\D{{0,12}}{NUM_PAT}\s*g",
    "dietary_fiber_g":       rf"(?:dietary\s*)?fiber\D{{0,12}}{NUM_PAT}\s*g",
    "total_sugars_g":        rf"(?:total\s*)?sugars\D{{0,12}}{NUM_PAT}\s*g",
    "added_sugars_g":        rf"added\s*sugars\D{{0,12}}{NUM_PAT}\s*g",
    "protein_g":             rf"protein\D{{0,12}}{NUM_PAT}\s*g",
}

def extract_nutrition_ocr_pytesseract(pil_img):
    if not TESS_OK:
        return None, "pytesseract is not installed/available on this server."
    text = pytesseract.image_to_string(pil_img)
    low = text.lower().replace("\n", " ")
    result = {"product_name": None, "serving_size": None, "servings_per_container": None}
    for key, pattern in OCR_PATTERNS.items():
        m = re.search(pattern, low)
        result[key] = float(m.group(1)) if m else None
    sm = re.search(r"serving size\D{0,20}([\d\.]+\s*\w+\s*\(?\d*\s*g?\)?)", low)
    if sm:
        result["serving_size"] = sm.group(1)
    result["raw_text"] = text
    return result, None

def extract_nutrition_ocr_hf(pil_img):
    if not (HF_CLIENT_OK and HF_TOKEN):
        return None, "HF Inference client / HF_TOKEN not available."
    client = InferenceClient(model=CFG.OCR_MODEL, token=HF_TOKEN)
    # convert to jpeg base64 data: URI
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    # Using chat_completion with image+prompt (depends on model capabilities)
    prompt = ("Read the Nutrition Facts label image and return STRICT JSON with keys: "
              "product_name, serving_size, servings_per_container, calories, total_fat_g, "
              "saturated_fat_g, trans_fat_g, cholesterol_mg, sodium_mg, total_carbohydrates_g, "
              "dietary_fiber_g, total_sugars_g, added_sugars_g, protein_g. Use null for unreadable values.")
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}
    ]
    try:
        resp = client.chat_completion(messages=messages, max_tokens=512, temperature=0.0)
        raw = ""
        try:
            raw = resp.choices[0].message.content
        except Exception:
            raw = resp.choices[0].message["content"]
        # try to extract JSON substring
        m = re.search(r"(\{[\s\S]*\})", raw)
        if not m:
            # maybe raw is already JSON-like
            raw_text = raw.strip()
        else:
            raw_text = m.group(1)
        data = json.loads(raw_text)
        return data, None
    except Exception as e:
        return None, f"OCR HF API error: {e}"

def extract_nutrition_ocr(pil_img):
    # prefer HF OCR if token & client available (more robust), otherwise pytesseract
    if HF_TOKEN and HF_CLIENT_OK:
        data, err = extract_nutrition_ocr_hf(pil_img)
        if data is not None:
            return data, None
        # fallback to pytesseract if HF fails
    if TESS_OK:
        return extract_nutrition_ocr_pytesseract(pil_img)
    return None, "No OCR method available (install pytesseract or provide HF_TOKEN)."

# ----------------------------
# Chat router & assistant
# ----------------------------
class Route:
    def __init__(self, intent, tool, confidence):
        self.intent, self.tool, self.confidence = intent, tool, confidence

class AIRouter:
    def __init__(self):
        self.rules = {
            "daily_summary": [r"how many calories.*(left|remaining)", r"calories.*(left|remaining)",
                               r"what did i eat today", r"daily.*summary", r"today.*nutrition"],
            "remaining_macros": [r"how much protein.*(left|remaining)", r"how much.*protein.*need",
                                  r"how much.*carb.*(left|remaining)", r"how much.*fat.*(left|remaining)",
                                  r"remaining.*(protein|carb|fat|macro)"],
            "recommendation": [r"what should i eat", r"what can i eat", r"recommend.*meal", r"recommend.*food"],
            "food_comparison": [r"compare", r"which.*better", r"better.*between"],
            "user_targets": [r"my calorie target", r"my protein target", r"my macros"],
        }

    def route(self, message):
        text = re.sub(r"\s+", " ", message.lower().strip())
        scores = {i: sum(bool(re.search(p, text)) for p in ps) for i, ps in self.rules.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Route("general_nutrition_chat", None, 0.25)
        return Route(best, best, min(0.95, 0.55 + scores[best] * 0.15))

ROUTER = AIRouter()

SYSTEM_PROMPT = """You are an AI Nutrition Coach inside a Live Nutrition app.
Use ONLY the trusted nutrition data provided to you. Never invent numbers.
Be concise, supportive, never shame the user for any food choice, and never claim to be a doctor.
Recommend consulting a professional for medical questions."""

@st.cache_resource(show_spinner=False)
def get_hf_client(model_id):
    if not HF_TOKEN:
        return None
    if not HF_CLIENT_OK:
        return None
    try:
        return InferenceClient(model=model_id, token=HF_TOKEN)
    except Exception:
        return None

def rule_based_reply(intent, message, ctx):
    r = ctx["remaining"]
    if intent == "daily_summary":
        return (f"Today you've had {ctx['consumed_today']['calories']:.0f} kcal, "
                f"{ctx['consumed_today']['protein']:.0f}g protein. You have {r['calories']:.0f} kcal left.")
    if intent == "remaining_macros":
        return f"Remaining → {r['calories']:.0f} kcal, {r['protein']:.0f}g protein, {r['carb']:.0f}g carbs, {r['fat']:.0f}g fat."
    if intent == "user_targets":
        t = ctx["daily_target"]
        return f"Targets ({ctx['goal']}): {t['calories']} kcal, {t['protein']}g protein, {t['carb']}g carbs, {t['fat']}g fat."
    if intent == "recommendation":
        return (f"With ~{r['calories']:.0f} kcal left and {r['protein']:.0f}g protein, "
                "a lean-protein meal with veggies and a healthy carb fits well.")
    return ("I can help with calories, macros, meal suggestions, and comparisons — ask me anything!")

def llm_reply(message, ctx):
    if not HF_TOKEN or not HF_CLIENT_OK:
        return None
    client = get_hf_client(CFG.CHAT_MODEL)
    if not client:
        return None
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nTrusted data:\n" + json.dumps(ctx)},
            {"role": "user", "content": message},
        ]
        resp = client.chat_completion(messages=messages, max_tokens=400, temperature=0.7)
        try:
            return resp.choices[0].message["content"]
        except Exception:
            try:
                return resp.choices[0].message.content
            except Exception:
                return None
    except Exception:
        return None

def chatbot_reply(message):
    ctx = build_chat_context()
    route = ROUTER.route(message)
    reply = llm_reply(message, ctx)
    if not reply:
        reply = rule_based_reply(route.intent, message, ctx)
    return reply

# ----------------------------
# Session state and helpers
# ----------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("meal_log", [])  # each entry: dict with date,time,food,total_mass,total_calories,total_protein,total_carb,total_fat,source
    ss.setdefault("shopping_list", [])
    ss.setdefault("chat_history", [])
    ss.setdefault("targets", {"calories":2000,"protein":130,"carb":230,"fat":65})
    ss.setdefault("goal", "Maintain Weight")
    ss.setdefault("last_scan", None)
    ss.setdefault("last_scan_name", "Scanned Food")
    ss.setdefault("last_ocr", None)
    ss.setdefault("use_midas", False)
    ss.setdefault("midas_loaded", False)

def add_to_log(name, nutrition, source="scanner"):
    # normalize nutrition dict keys and append to meal_log
    mass = (nutrition.get("total_mass")
            or nutrition.get("mass")
            or nutrition.get("portion_g")
            or nutrition.get("serving_size")
            or None)
    calories = (nutrition.get("total_calories") or nutrition.get("calories") or nutrition.get("cal") or 0.0)
    protein = (nutrition.get("total_protein") or nutrition.get("protein") or nutrition.get("protein_g") or 0.0)
    carb = (nutrition.get("total_carb") or nutrition.get("total_carbohydrates_g") or nutrition.get("carb") or nutrition.get("carbs") or 0.0)
    fat = (nutrition.get("total_fat") or nutrition.get("total_fat_g") or nutrition.get("fat") or 0.0)
    entry = {
        "date": date.today().isoformat(),
        "time": time.strftime("%H:%M:%S"),
        "food": name,
        "total_mass": float(mass) if mass else None,
        "total_calories": float(calories),
        "total_protein": float(protein),
        "total_carb": float(carb),
        "total_fat": float(fat),
        "source": source
    }
    st.session_state.meal_log.append(entry)

def totals_today():
    if not st.session_state.meal_log:
        return {"calories":0,"protein":0,"carb":0,"fat":0}
    df = pd.DataFrame(st.session_state.meal_log)
    today = date.today().isoformat()
    df = df[df["date"]==today] if "date" in df.columns else df
    return {
        "calories": df["total_calories"].sum() if "total_calories" in df.columns else 0,
        "protein": df["total_protein"].sum() if "total_protein" in df.columns else 0,
        "carb": df["total_carb"].sum() if "total_carb" in df.columns else 0,
        "fat": df["total_fat"].sum() if "total_fat" in df.columns else 0,
    }

def build_chat_context():
    tot = totals_today()
    tgt = st.session_state.targets
    return {
        "goal": st.session_state.goal,
        "daily_target": tgt,
        "consumed_today": tot,
        "remaining": {k: round(tgt[k]-tot[k],1) for k in ["calories","protein","carb","fat"]},
        "meal_log": st.session_state.meal_log[-8:],
    }

# ----------------------------
# Sidebar: targets & profile
# ----------------------------
with st.sidebar:
    st.markdown("### 🎯 Your Goal & Daily Targets")
    st.session_state.goal = st.selectbox("Goal", ["Lose Weight","Maintain Weight","Gain Muscle"], index=["Lose Weight","Maintain Weight","Gain Muscle"].index(st.session_state.goal if "goal" in st.session_state else "Maintain Weight"))
    with st.expander("🧮 Auto-calculate targets (Mifflin-St Jeor)"):
        sex = st.radio("Sex", ["Male","Female"], horizontal=True)
        age = st.number_input("Age", 10, 90, 28)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        height = st.number_input("Height (cm)", 120, 220, 170)
        activity = st.select_slider("Activity level", options=[1.2,1.375,1.55,1.725,1.9], value=1.55,
                                     format_func=lambda x: {1.2:"Sedentary",1.375:"Light",1.55:"Moderate",1.725:"Active",1.9:"Very Active"}[x])
        if st.button("⚡ Calculate Targets", use_container_width=True):
            bmr = 10*weight + 6.25*height - 5*age + (5 if sex=="Male" else -161)
            tdee = bmr * activity
            if st.session_state.goal == "Lose Weight": tdee *= 0.8
            elif st.session_state.goal == "Gain Muscle": tdee *= 1.15
            protein = weight * (2.0 if st.session_state.goal=="Gain Muscle" else 1.6)
            fat = (tdee * 0.25) / 9
            carb = (tdee - protein*4 - fat*9) / 4
            st.session_state.targets = {"calories": round(tdee), "protein": round(protein),
                                         "carb": round(max(carb,0)), "fat": round(fat)}
            st.success("Targets updated!")

    st.markdown("#### Manual Overrides")
    t = st.session_state.targets
    cals = st.number_input("Calories target", 500, 6000, int(t.get("calories",2000)))
    prot = st.number_input("Protein target (g)", 10, 400, int(t.get("protein",130)))
    carbs = st.number_input("Carbs target (g)", 10, 600, int(t.get("carb",230)))
    fats = st.number_input("Fat target (g)", 10, 300, int(t.get("fat",65)))
    st.session_state.targets = {"calories":int(cals),"protein":int(prot),"carb":int(carbs),"fat":int(fats)}

    st.divider()
    st.markdown("Model & APIs")
    st.write(f"Model: `{CFG.HF_REPO}/{CFG.HF_FILE}`")
    st.write("HF Token: " + ("✅ Provided" if HF_TOKEN else "❌ Missing"))
    st.checkbox("Use MiDaS depth (experimental)", value=st.session_state.use_midas, key="use_midas")
    if st.session_state.use_midas and not st.session_state.midas_loaded:
        with st.spinner("Attempting to load MiDaS (this may take 30s+ on first run)..."):
            try:
                midas_tuple = try_load_midas()
                if midas_tuple[0] is not None:
                    st.session_state.midas_loaded = True
                    st.success("MiDaS loaded — depth estimation enabled.")
                else:
                    st.session_state.midas_loaded = False
                    st.warning("MiDaS could not be loaded; falling back to pseudo-depth.")
            except Exception as e:
                st.session_state.midas_loaded = False
                st.warning(f"MiDaS load failed: {e}")
    st.caption("Add HF_TOKEN in Streamlit Secrets to enable model download (if private) and full LLM / OCR features.")

# ----------------------------
# Tabs / Pages
# ----------------------------
tab_scan, tab_ocr, tab_dash, tab_shop, tab_chat = st.tabs(["📷 Live Scan","🧾 Label OCR","📊 Dashboard","🛒 Shopping & Compare","🤖 AI Coach"])

# TAB: Live Scan
with tab_scan:
    st.markdown("### 📸 Live Camera — Scan your meal")
    col1, col2 = st.columns([1,1])
    with col1:
        cam_img = st.camera_input("Use your camera", key="scan_cam")
        upl_img = st.file_uploader("...or upload a photo", type=["jpg","png","jpeg"], key="scan_upl")
        food_name = st.text_input("Food name (optional)", placeholder="e.g., Salmon Bowl")

    image_to_use = None
    if cam_img is not None:
        image_to_use = Image.open(cam_img)
    elif upl_img is not None:
        image_to_use = Image.open(upl_img)

    with col2:
        if image_to_use is not None:
            st.image(image_to_use, caption="Captured", use_container_width=True)
            if st.button("🔍 Analyze Nutrition", type="primary", use_container_width=True):
                use_midas = st.session_state.use_midas and st.session_state.midas_loaded
                midas_tuple = try_load_midas() if use_midas else (None,None)
                try:
                    with st.spinner("Running model inference..."):
                        result = predict_nutrition(image_to_use, use_midas=use_midas, midas_tuple=midas_tuple)
                    st.session_state["last_scan"] = result
                    st.session_state["last_scan_name"] = food_name or "Scanned Food"
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Inference failed: {e}")

    if st.session_state.get("last_scan", None):
        r = st.session_state["last_scan"]
        st.markdown("### 🥘 Estimated Nutrition")
        cols = st.columns(5)
        labels = [("total_mass","Mass (g)"), ("total_calories","Calories"),
                  ("total_protein","Protein (g)"), ("total_carb","Carbs (g)"), ("total_fat","Fat (g)")]
        for c,(k,l) in zip(cols, labels):
            val = r.get(k, 0.0)
            c.markdown(f'<div class="card" style="text-align:center;"><h3 style="color:#2E7D32;margin:0;">{val:.1f}</h3><p style="margin:0;">{l}</p></div>', unsafe_allow_html=True)
        if st.button("➕ Add to Meal Log", use_container_width=True):
            add_to_log(st.session_state.get("last_scan_name","Scanned Food"), r, source="camera_ai")
            st.success("Added to meal log ✅")
            st.balloons()

# TAB: Label OCR
with tab_ocr:
    st.markdown("### 🧾 Nutrition Label OCR")
    colA, colB = st.columns(2)
    with colA:
        label_cam = st.camera_input("Capture label", key="ocr_cam")
        label_upl = st.file_uploader("...or upload label image", type=["jpg","jpeg","png"], key="ocr_upl")
    label_img = None
    if label_cam is not None:
        label_img = Image.open(label_cam)
    elif label_upl is not None:
        label_img = Image.open(label_upl)

    with colB:
        if label_img is not None:
            st.image(label_img, caption="Label", use_container_width=True)
            if st.button("📖 Extract Nutrition Data", type="primary"):
                with st.spinner("Reading label..."):
                    data, err = extract_nutrition_ocr(label_img)
                if err:
                    st.error(err)
                else:
                    st.session_state["last_ocr"] = data
                    st.success("Label extracted!")

    if st.session_state.get("last_ocr", None):
        data = st.session_state["last_ocr"]
        show = {k:v for k,v in data.items() if k not in ["raw_text"]}
        st.dataframe(pd.DataFrame(list(show.items()), columns=["Nutrient","Value"]), use_container_width=True)
        prod_name = st.text_input("Product name for log", value=data.get("product_name") or "Packaged Food")
        if st.button("➕ Add Label Item to Meal Log"):
            add_to_log(prod_name, data, source="label_ocr")
            st.success("Added to meal log ✅")

# TAB: Dashboard
with tab_dash:
    st.markdown("### 📆 Today's Summary")
    tot = totals_today()
    tgt = st.session_state.targets

    def pill(label, consumed, target, unit=""):
        remaining = max(target - consumed, 0)
        st.markdown(f"**{label}** — {consumed:.0f}/{target}{unit}  (remaining: {remaining:.0f}{unit})")
        pct = min(100, (consumed/target*100) if target else 0)
        st.markdown(f'<div style="background:#e8f5e9;border-radius:12px;height:14px;"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#66BB6A,#2E7D32);border-radius:12px;"></div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pill("🔥 Calories", tot["calories"], tgt["calories"], unit=" kcal")
        pill("💪 Protein", tot["protein"], tgt["protein"], unit=" g")
    with c2:
        pill("🍞 Carbs", tot["carb"], tgt["carb"], unit=" g")
        pill("🥑 Fat", tot["fat"], tgt["fat"], unit=" g")

    st.markdown("### 🍽️ Meal Log")
    if st.session_state.meal_log:
        df = pd.DataFrame(st.session_state.meal_log)
        st.dataframe(df, use_container_width=True)
        if st.button("🗑️ Clear Meal Log"):
            st.session_state.meal_log = []
            st.experimental_rerun()
    else:
        st.info("No meals logged yet — scan a food or nutrition label to get started!")

    st.markdown("### 💡 Smart Meal Suggestions")
    remaining_cal = tgt["calories"] - tot["calories"]
    SUGGESTIONS = {
        "Lose Weight": ["Grilled chicken breast + steamed veggies", "Greek yogurt with berries", "Egg white omelet + spinach"],
        "Maintain Weight": ["Salmon bowl with rice & avocado", "Turkey wrap with whole wheat", "Vegetable stir-fry with tofu"],
        "Gain Muscle": ["Chicken, rice & broccoli bowl", "Peanut butter banana smoothie + whey", "Beef steak with sweet potato"],
    }
    if remaining_cal <= 0:
        st.warning("You've hit your calorie target for today — great job! Light, protein-rich foods are best if still hungry.")
    else:
        for s in SUGGESTIONS.get(st.session_state.goal, SUGGESTIONS["Maintain Weight"]):
            st.markdown(f'<div class="card">🍴 {s} <span class="metric-pill">~{int(remaining_cal/3)} kcal portion</span></div>', unsafe_allow_html=True)

# TAB: Shopping & Compare
with tab_shop:
    st.markdown("### 🛒 Smart Shopping List")
    new_item = st.text_input("Add an item to your shopping list")
    if st.button("➕ Add Item") and new_item:
        st.session_state.shopping_list.append(new_item)
    if st.session_state.shopping_list:
        for i, item in enumerate(st.session_state.shopping_list):
            c1, c2 = st.columns([6,1])
            c1.write(f"• {item}")
            if c2.button("❌", key=f"del_{i}"):
                st.session_state.shopping_list.pop(i)
                st.experimental_rerun()
    else:
        st.info("Your shopping list is empty.")

    st.markdown("---")
    st.markdown("### ⚖️ Compare Two Foods / Products")
    colX, colY = st.columns(2)
    def manual_entry(prefix):
        st.markdown(f"**{prefix}**")
        name = st.text_input(f"{prefix} name", key=f"{prefix}_name")
        cal = st.number_input(f"{prefix} calories", 0, 3000, 0, key=f"{prefix}_cal")
        pro = st.number_input(f"{prefix} protein (g)", 0, 200, 0, key=f"{prefix}_pro")
        carb = st.number_input(f"{prefix} carbs (g)", 0, 300, 0, key=f"{prefix}_carb")
        fat = st.number_input(f"{prefix} fat (g)", 0, 200, 0, key=f"{prefix}_fat")
        sugar = st.number_input(f"{prefix} sugar (g)", 0, 200, 0, key=f"{prefix}_sugar")
        return {"name": name or prefix, "calories": cal, "protein": pro, "carb": carb, "fat": fat, "sugar": sugar}

    with colX: A = manual_entry("Product A")
    with colY: B = manual_entry("Product B")

    if st.button("⚖️ Compare", type="primary"):
        comp_df = pd.DataFrame([A, B]).set_index("name")
        st.dataframe(comp_df, use_container_width=True)
        healthier = A["name"] if (A["calories"], A["sugar"]) <= (B["calories"], B["sugar"]) else B["name"]
        st.success(f"✅ **{healthier}** looks like the better choice (lower calories/sugar).")

# TAB: Chat / AI Coach
with tab_chat:
    st.markdown("### 🤖 Ask Your AI Nutrition Coach")
    for msg in st.session_state.chat_history:
        cls = "chat-user" if msg["role"]=="user" else "chat-bot"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

    user_msg = st.chat_input("Ask about your meals, macros, or get suggestions...")

    with st.expander("🎙️ Voice Input (optional)"):
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("Record your question")
        else:
            audio_val = None
            st.info("Your Streamlit version does not support st.audio_input in this environment.")
        if audio_val is not None and SR_OK and st.button("🎤 Transcribe & Send"):
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(audio_val.getvalue())) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                user_msg = text
                st.info(f"Heard: {text}")
            except Exception as e:
                st.error(f"Could not transcribe audio: {e}")

    if user_msg:
        st.session_state.chat_history.append({"role":"user","content":user_msg})
        with st.spinner("Thinking..."):
            reply = chatbot_reply(user_msg)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.experimental_rerun()

    # TTS for last bot message
    if GTTS_OK and any(m["role"]=="assistant" for m in st.session_state.chat_history):
        last_bot = [m for m in st.session_state.chat_history if m["role"]=="assistant"][-1]["content"]
        if st.button("🔊 Play Last Response"):
            try:
                tts = gTTS(last_bot[:500], lang="en")
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf.getvalue(), format="audio/mp3")
            except Exception as e:
                st.error(f"TTS failed: {e}")

st.markdown("---")
st.caption("⚠️ This assistant provides general nutrition guidance, not medical advice. Consult a professional for health concerns.")
# Initialize session state
init_state()
