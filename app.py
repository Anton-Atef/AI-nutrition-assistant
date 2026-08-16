# =========================================================
#  🥗 AI NUTRITION & SMART SHOPPING ASSISTANT — Streamlit
#  Model weights streamed from Hugging Face Hub
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="AI Nutrition & Smart Shopping Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 🎨 GLOBAL CSS / ANIMATIONS
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated gradient header */
.main-header {
    background: linear-gradient(-45deg, #4CAF50, #8BC34A, #2E7D32, #66BB6A);
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
    padding: 28px 20px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.main-header h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
.main-header p { margin: 4px 0 0; opacity: 0.95; }

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.85);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.10);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(76,175,80,0.15);
}
.glass-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 12px 28px rgba(76,175,80,0.28);
}

/* Metric pill */
.metric-pill {
    display:inline-block; padding:6px 14px; border-radius:20px;
    background: linear-gradient(90deg,#43A047,#66BB6A);
    color:white; font-weight:600; font-size:0.85rem; margin:3px;
    animation: pop 0.4s ease;
}
@keyframes pop { from{transform:scale(0.7); opacity:0;} to{transform:scale(1); opacity:1;} }

/* Progress bar animation */
.progress-wrap { background:#e8f5e9; border-radius:12px; height:18px; overflow:hidden; margin-bottom:6px;}
.progress-fill {
    height:100%; border-radius:12px;
    background: linear-gradient(90deg,#66BB6A,#2E7D32);
    animation: growBar 1.2s ease-out;
}
@keyframes growBar { from{width:0%;} }

/* Chat bubble */
.chat-user { background:#DCF8C6; padding:10px 14px; border-radius:14px 14px 0 14px; margin:6px 0; display:inline-block; max-width:80%; float:right; clear:both;}
.chat-bot  { background:#F1F0F0; padding:10px 14px; border-radius:14px 14px 14px 0; margin:6px 0; display:inline-block; max-width:80%; float:left; clear:both;}

/* Pulse for live camera */
.pulse-dot {
    height:12px; width:12px; background:#e53935; border-radius:50%;
    display:inline-block; margin-right:6px;
    animation: pulse 1.4s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(229,57,53,0.6);}
    70% { box-shadow: 0 0 0 10px rgba(229,57,53,0);}
    100% { box-shadow: 0 0 0 0 rgba(229,57,53,0);}
}

footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🥗 Live AI Nutrition & Smart Shopping Assistant</h1>
  <p>Camera-based food recognition • Nutrition Facts OCR • Macro tracking • AI Coach • Voice</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 📦 IMPORTS
# ---------------------------------------------------------
import os, io, re, json, time, base64, random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import timm
from huggingface_hub import hf_hub_download

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

# ---------------------------------------------------------
# ⚙️ CONFIG
# ---------------------------------------------------------
class CFG:
    HF_REPO   = "Anton-Atef/AI-nutrition-assistant"
    HF_FILE   = "best_nutrition_rgbd.pt"
    BACKBONE  = "convnext_small"
    IN_CHANS  = 4
    IMG_SIZE  = 256
    TARGET_COLS = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def get_hf_token():
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        pass
    return token or os.environ.get("HF_TOKEN")

# ---------------------------------------------------------
# 🧠 MODEL DEFINITION (must match training)
# ---------------------------------------------------------
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
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource(show_spinner="🔽 Downloading AI model from Hugging Face Hub (first run only)...")
def load_model():
    token = get_hf_token()
    ckpt_path = hf_hub_download(repo_id=CFG.HF_REPO, filename=CFG.HF_FILE, token=token)
    ckpt = torch.load(ckpt_path, map_location=CFG.DEVICE)

    backbone = ckpt.get("backbone", CFG.BACKBONE)
    in_chans = ckpt.get("in_chans", CFG.IN_CHANS)
    target_cols = ckpt.get("target_cols", CFG.TARGET_COLS)

    model = NutritionNet(backbone, in_chans, len(target_cols))
    model.load_state_dict(ckpt["model"])
    model.eval().to(CFG.DEVICE)

    y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
    y_std  = np.array(ckpt["y_std"], dtype=np.float32)
    return model, y_mean, y_std, target_cols

# ---------------------------------------------------------
# 🖼️ PREPROCESS + PREDICT
# ---------------------------------------------------------
def preprocess_image(pil_img, img_size=CFG.IMG_SIZE):
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    rgb_r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    top, left = (img_size - nh) // 2, (img_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = rgb_r

    rgb_norm = canvas.astype(np.float32) / 255.0
    rgb_norm = (rgb_norm - IMAGENET_MEAN) / IMAGENET_STD

    # NOTE: no physical depth sensor on phone/webcam ->
    # neutral pseudo-depth channel (mid-range), matches training normalization (0.5 -> 0 after scaling)
    depth = np.full((img_size, img_size, 1), 0.5, dtype=np.float32)
    depth = (depth - 0.5) / 0.25

    img4 = np.concatenate([rgb_norm, depth], axis=2).astype(np.float32)
    x = torch.from_numpy(img4).permute(2, 0, 1).unsqueeze(0)
    return x

def predict_nutrition(pil_img):
    model, y_mean, y_std, target_cols = load_model()
    x = preprocess_image(pil_img).to(CFG.DEVICE)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    pred = pred * y_std + y_mean
    pred = np.clip(pred, 0, None)
    return dict(zip(target_cols, pred.tolist()))

# ---------------------------------------------------------
# 🧾 NUTRITION LABEL OCR  (pytesseract based, lightweight)
# ---------------------------------------------------------
NUM_PAT = r"([\d]+\.?[\d]*)"

OCR_PATTERNS = {
    "calories":              rf"calories\D{{0,10}}{NUM_PAT}",
    "total_fat_g":           rf"total\s*fat\D{{0,10}}{NUM_PAT}\s*g",
    "saturated_fat_g":       rf"saturated\s*fat\D{{0,10}}{NUM_PAT}\s*g",
    "trans_fat_g":           rf"trans\s*fat\D{{0,10}}{NUM_PAT}\s*g",
    "cholesterol_mg":        rf"cholesterol\D{{0,10}}{NUM_PAT}\s*mg",
    "sodium_mg":             rf"sodium\D{{0,10}}{NUM_PAT}\s*mg",
    "total_carbohydrates_g": rf"total\s*carb\w*\D{{0,10}}{NUM_PAT}\s*g",
    "dietary_fiber_g":       rf"(?:dietary\s*)?fiber\D{{0,10}}{NUM_PAT}\s*g",
    "total_sugars_g":        rf"(?:total\s*)?sugars\D{{0,10}}{NUM_PAT}\s*g",
    "added_sugars_g":        rf"added\s*sugars\D{{0,10}}{NUM_PAT}\s*g",
    "protein_g":             rf"protein\D{{0,10}}{NUM_PAT}\s*g",
}

def extract_nutrition_ocr(pil_img):
    if not TESS_OK:
        return None, "Tesseract not available on this server."
    text = pytesseract.image_to_string(pil_img)
    low = text.lower().replace("\n", " ")
    result = {"product_name": None, "serving_size": None, "servings_per_container": None}
    for key, pattern in OCR_PATTERNS.items():
        m = re.search(pattern, low)
        result[key] = float(m.group(1)) if m else None
    sm = re.search(r"serving size\D{0,15}([\d\.]+\s*\w+\s*\(?\d*\s*g?\)?)", low)
    if sm:
        result["serving_size"] = sm.group(1)
    result["raw_text"] = text
    return result, None

# ---------------------------------------------------------
# 🧭 INTENT ROUTER (for the chatbot)
# ---------------------------------------------------------
class Route:
    def __init__(self, intent, tool, confidence):
        self.intent, self.tool, self.confidence = intent, tool, confidence

class AIRouter:
    def __init__(self):
        self.rules = {
            "daily_summary": [r"how many calories.*(left|remaining)", r"calories.*(left|remaining)",
                               r"what did i eat today", r"how am i doing today", r"daily.*summary", r"today.*nutrition"],
            "remaining_macros": [r"how much protein.*(left|remaining)", r"how much.*protein.*need",
                                  r"how much.*carb.*(left|remaining)", r"how much.*fat.*(left|remaining)",
                                  r"remaining.*(protein|carb|fat|macro)"],
            "recommendation": [r"what should i eat", r"what can i eat", r"recommend.*meal", r"recommend.*food",
                                r"what.*eat.*dinner", r"what.*eat.*lunch", r"what.*eat.*breakfast", r"give me.*meal"],
            "food_comparison": [r"compare", r"which.*better", r"better.*between", r"difference between"],
            "user_targets": [r"my calorie target", r"my protein target", r"my macros", r"what.*my.*target"],
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

# ---------------------------------------------------------
# 🗂️ SESSION STATE
# ---------------------------------------------------------
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

def totals_today():
    if not st.session_state.meal_log:
        return {"calories": 0, "protein": 0, "carb": 0, "fat": 0}
    df = pd.DataFrame(st.session_state.meal_log)
    return {
        "calories": df["total_calories"].sum(),
        "protein": df["total_protein"].sum(),
        "carb": df["total_carb"].sum(),
        "fat": df["total_fat"].sum(),
    }

def add_to_log(name, nutrition):
    st.session_state.meal_log.append({
        "time": time.strftime("%H:%M:%S"),
        "food": name,
        "total_mass": nutrition.get("total_mass", 0),
        "total_calories": nutrition.get("total_calories", nutrition.get("calories", 0)),
        "total_fat": nutrition.get("total_fat", nutrition.get("total_fat_g", 0)),
        "total_carb": nutrition.get("total_carb", nutrition.get("total_carbohydrates_g", 0)),
        "total_protein": nutrition.get("total_protein", nutrition.get("protein_g", 0)),
    })

# ---------------------------------------------------------
# 🎯 SIDEBAR — Goals & Targets
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Your Goal & Targets")
    st.session_state.goal = st.selectbox("Goal", ["Lose Weight", "Maintain Weight", "Gain Muscle"],
                                          index=["Lose Weight","Maintain Weight","Gain Muscle"].index(st.session_state.goal))

    with st.expander("🧮 Auto-calculate targets (Mifflin-St Jeor)"):
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        age = st.number_input("Age", 10, 90, 25)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        height = st.number_input("Height (cm)", 120, 220, 170)
        activity = st.select_slider("Activity level", options=[1.2,1.375,1.55,1.725,1.9], value=1.55,
                                     format_func=lambda x: {1.2:"Sedentary",1.375:"Light",1.55:"Moderate",
                                                             1.725:"Active",1.9:"Very Active"}[x])
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

    st.markdown("#### Manual Override")
    t = st.session_state.targets
    t["calories"] = st.number_input("Calories target", 500, 6000, int(t["calories"]))
    t["protein"]  = st.number_input("Protein target (g)", 10, 400, int(t["protein"]))
    t["carb"]     = st.number_input("Carbs target (g)", 10, 600, int(t["carb"]))
    t["fat"]      = st.number_input("Fat target (g)", 10, 300, int(t["fat"]))

    st.divider()
    st.caption("🔑 Add `HF_TOKEN` in Streamlit **Secrets** to enable the private model + full LLM chat.")

# ---------------------------------------------------------
# 🧭 TABS
# ---------------------------------------------------------
tab_scan, tab_ocr, tab_dash, tab_shop, tab_chat = st.tabs(
    ["📷 Live Scan", "🧾 Label OCR", "📊 Dashboard", "🛒 Shopping & Compare", "🤖 AI Coach"]
)

# =========================================================
# 📷 TAB 1 — LIVE SCAN
# =========================================================
with tab_scan:
    st.markdown('<span class="pulse-dot"></span> **Live Camera — Point at your food**', unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])

    with col1:
        cam_img = st.camera_input("Capture your meal", key="scan_cam")
        upl_img = st.file_uploader("...or upload a food photo", type=["jpg","jpeg","png"], key="scan_upl")
        food_name = st.text_input("Food name (optional)", placeholder="e.g., Salmon Bowl")

    image_to_use = None
    if cam_img is not None:
        image_to_use = Image.open(cam_img)
    elif upl_img is not None:
        image_to_use = Image.open(upl_img)

    with col2:
        if image_to_use is not None:
            st.image(image_to_use, caption="Captured Frame", use_container_width=True)
            if st.button("🔍 Analyze Nutrition", type="primary", use_container_width=True):
                with st.spinner("Running Nutrition5k model..."):
                    result = predict_nutrition(image_to_use)
                st.session_state["last_scan"] = result
                st.session_state["last_scan_name"] = food_name or "Scanned Food"

    if "last_scan" in st.session_state:
        r = st.session_state["last_scan"]
        st.markdown("### 🥘 Estimated Nutrition")
        cols = st.columns(5)
        labels = [("total_mass","Mass (g)"), ("total_calories","Calories"),
                  ("total_protein","Protein (g)"), ("total_carb","Carbs (g)"), ("total_fat","Fat (g)")]
        for c,(k,l) in zip(cols, labels):
            c.markdown(f"""<div class="glass-card" style="text-align:center;">
                <h3 style="color:#2E7D32;margin:0;">{r[k]:.1f}</h3><p style="margin:0;">{l}</p></div>""",
                unsafe_allow_html=True)

        if st.button("➕ Add to Meal Log", use_container_width=True):
            add_to_log(st.session_state.get("last_scan_name","Scanned Food"), r)
            st.success(f"Added '{st.session_state['last_scan_name']}' to your meal log! ✅")
            st.balloons()

# =========================================================
# 🧾 TAB 2 — NUTRITION LABEL OCR
# =========================================================
with tab_ocr:
    st.markdown("Scan a packaged food's **Nutrition Facts** table.")
    colA, colB = st.columns(2)
    with colA:
        label_cam = st.camera_input("Capture label", key="ocr_cam")
        label_upl = st.file_uploader("...or upload label image", type=["jpg","jpeg","png"], key="ocr_upl")

    label_img = None
    if label_cam is not None: label_img = Image.open(label_cam)
    elif label_upl is not None: label_img = Image.open(label_upl)

    with colB:
        if label_img is not None:
            st.image(label_img, caption="Nutrition Label", use_container_width=True)
            if st.button("📖 Extract Nutrition Data", type="primary"):
                with st.spinner("Reading label with OCR..."):
                    data, err = extract_nutrition_ocr(label_img)
                if err:
                    st.error(err)
                else:
                    st.session_state["last_ocr"] = data

    if "last_ocr" in st.session_state:
        data = st.session_state["last_ocr"]
        show = {k:v for k,v in data.items() if k not in ["raw_text"]}
        st.dataframe(pd.DataFrame(list(show.items()), columns=["Nutrient","Value"]), use_container_width=True)
        prod_name = st.text_input("Product name for log", value=data.get("product_name") or "Packaged Food")
        if st.button("➕ Add Label Item to Meal Log"):
            add_to_log(prod_name, data)
            st.success("Added to meal log! ✅")

# =========================================================
# 📊 TAB 3 — DASHBOARD
# =========================================================
with tab_dash:
    st.markdown("### 📆 Today's Summary")
    tot = totals_today()
    tgt = st.session_state.targets

    def bar(label, consumed, target, color_from="#66BB6A", color_to="#2E7D32", unit=""):
        pct = min(100, (consumed/target*100) if target else 0)
        remaining = max(target - consumed, 0)
        st.markdown(f"**{label}** — {consumed:.0f}/{target}{unit}  (remaining: {remaining:.0f}{unit})")
        st.markdown(f"""<div class="progress-wrap"><div class="progress-fill" style="width:{pct}%;"></div></div>""",
                    unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        bar("🔥 Calories", tot["calories"], tgt["calories"], unit=" kcal")
        bar("💪 Protein", tot["protein"], tgt["protein"], unit=" g")
    with c2:
        bar("🍞 Carbs", tot["carb"], tgt["carb"], unit=" g")
        bar("🥑 Fat", tot["fat"], tgt["fat"], unit=" g")

    st.markdown("### 🍽️ Meal Log")
    if st.session_state.meal_log:
        st.dataframe(pd.DataFrame(st.session_state.meal_log), use_container_width=True)
        if st.button("🗑️ Clear Meal Log"):
            st.session_state.meal_log = []
            st.rerun()
    else:
        st.info("No meals logged yet today — scan a food or nutrition label to get started!")

    st.markdown("### 💡 Smart Meal Suggestions")
    remaining_cal = tgt["calories"] - tot["calories"]
    remaining_protein = tgt["protein"] - tot["protein"]
    SUGGESTIONS = {
        "Lose Weight": ["Grilled chicken breast + steamed veggies", "Greek yogurt with berries", "Egg white omelet + spinach"],
        "Maintain Weight": ["Salmon bowl with rice & avocado", "Turkey wrap with whole wheat", "Vegetable stir-fry with tofu"],
        "Gain Muscle": ["Chicken, rice & broccoli bowl", "Peanut butter banana smoothie + whey", "Beef steak with sweet potato"],
    }
    if remaining_cal <= 0:
        st.warning("You've hit your calorie target for today — great job! Light, protein-rich foods are best if still hungry.")
    else:
        for s in SUGGESTIONS[st.session_state.goal]:
            st.markdown(f"""<div class="glass-card" style="margin-bottom:8px;">🍴 {s}
            <span class="metric-pill">~{int(remaining_cal/3)} kcal portion</span></div>""", unsafe_allow_html=True)

# =========================================================
# 🛒 TAB 4 — SHOPPING LIST & COMPARE
# =========================================================
with tab_shop:
    st.markdown("### 🛒 Smart Shopping List")
    new_item = st.text_input("Add an item to your shopping list")
    if st.button("➕ Add Item") and new_item:
        st.session_state.shopping_list.append(new_item)
    if st.session_state.shopping_list:
        for i, item in enumerate(st.session_state.shopping_list):
            c1, c2 = st.columns([5,1])
            c1.write(f"• {item}")
            if c2.button("❌", key=f"del_{i}"):
                st.session_state.shopping_list.pop(i)
                st.rerun()
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

# =========================================================
# 🤖 TAB 5 — AI CHATBOT (+ optional voice)
# =========================================================
with tab_chat:
    st.markdown("### 🤖 Ask Your AI Nutrition Coach")

    def build_context():
        tot = totals_today()
        tgt = st.session_state.targets
        return {
            "goal": st.session_state.goal,
            "daily_target": tgt,
            "consumed_today": tot,
            "remaining": {k: round(tgt[k]-tot[k],1) for k in ["calories","protein","carb","fat"]},
            "meal_log": st.session_state.meal_log,
        }

    def rule_based_reply(intent, message, ctx):
        r = ctx["remaining"]
        if intent == "daily_summary":
            return (f"Today you've had **{ctx['consumed_today']['calories']:.0f} kcal**, "
                    f"**{ctx['consumed_today']['protein']:.0f}g protein**, "
                    f"**{ctx['consumed_today']['carb']:.0f}g carbs**, **{ctx['consumed_today']['fat']:.0f}g fat**. "
                    f"You have **{r['calories']:.0f} kcal** left for today.")
        if intent == "remaining_macros":
            return f"Remaining today → 🔥{r['calories']:.0f} kcal, 💪{r['protein']:.0f}g protein, 🍞{r['carb']:.0f}g carbs, 🥑{r['fat']:.0f}g fat."
        if intent == "user_targets":
            t = ctx["daily_target"]
            return f"Your daily targets ({ctx['goal']}): {t['calories']} kcal, {t['protein']}g protein, {t['carb']}g carbs, {t['fat']}g fat."
        if intent == "recommendation":
            return (f"With about **{r['calories']:.0f} kcal** and **{r['protein']:.0f}g protein** left, "
                     "a lean-protein meal with veggies and a healthy carb source would fit nicely.")
        return ("I can help with your calories, macros, meal suggestions, and food comparisons — "
                "just ask! For medical concerns, please consult a healthcare professional.")

    def llm_reply(message, ctx):
        token = get_hf_token()
        if not (HF_CLIENT_OK and token):
            return None
        try:
            client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + "\n\nTrusted data:\n" + json.dumps(ctx)},
                {"role": "user", "content": message},
            ]
            resp = client.chat_completion(messages=messages, max_tokens=400, temperature=0.7)
            return resp.choices[0].message["content"]
        except Exception:
            return None

    def chatbot_reply(message):
        ctx = build_context()
        route = ROUTER.route(message)
        reply = llm_reply(message, ctx)
        if not reply:
            reply = rule_based_reply(route.intent, message, ctx)
        return reply

    # ---- chat history render ----
    for msg in st.session_state.chat_history:
        cls = "chat-user" if msg["role"] == "user" else "chat-bot"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('<div style="clear:both;"></div>', unsafe_allow_html=True)

    user_msg = st.chat_input("Ask about your meals, macros, or get suggestions...")

    # 🎙️ Voice input (best-effort)
    with st.expander("🎙️ Voice Input (optional)"):
        audio_val = st.audio_input("Record your question") if hasattr(st, "audio_input") else None
        if audio_val is not None and SR_OK and st.button("🎤 Transcribe & Send"):
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_val) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                user_msg = text
                st.info(f"Heard: {text}")
            except Exception as e:
                st.error(f"Could not transcribe audio: {e}")

    if user_msg:
        st.session_state.chat_history.append({"role":"user", "content": user_msg})
        with st.spinner("Thinking..."):
            reply = chatbot_reply(user_msg)
        st.session_state.chat_history.append({"role":"assistant", "content": reply})
        st.rerun()

    # 🔊 Voice output for last bot message
    if GTTS_OK and st.session_state.chat_history:
        last_bot = [m for m in st.session_state.chat_history if m["role"]=="assistant"]
        if last_bot and st.button("🔊 Play Last Response"):
            tts = gTTS(last_bot[-1]["content"])
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            st.audio(buf.getvalue(), format="audio/mp3")

st.markdown("---")
st.caption("⚠️ This assistant provides general nutrition guidance, not medical advice. Consult a professional for health concerns.")