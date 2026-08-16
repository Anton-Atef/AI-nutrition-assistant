import os, io, re, json, math, random, time
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 📦 OPTIONAL DEPENDENCY GUARDS
# =========================================================
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    cv2 = None

try:
    import pytesseract
    TESS_OK = True
except Exception:
    TESS_OK = False

try:
    import torch
    import torch.nn as nn
    import timm
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None

# =========================================================
# 🎨 PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Live AI Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 CUSTOM CSS + ANIMATIONS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); }

/* ---------- Animated gradient header ---------- */
.hero-header {
    background: linear-gradient(-45deg, #2e7d32, #66bb6a, #43a047, #81c784);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite;
    padding: 26px 24px;
    border-radius: 20px;
    color: white;
    box-shadow: 0 10px 30px rgba(46,125,50,0.35);
    margin-bottom: 18px;
    animation: gradientShift 10s ease infinite, fadeInDown 0.7s ease;
}
@keyframes gradientShift {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
@keyframes fadeInDown {
    from {opacity:0; transform:translateY(-16px);}
    to {opacity:1; transform:translateY(0);}
}
.hero-header h1 { margin:0; font-size:2rem; font-weight:800; }
.hero-header p { margin:4px 0 0; opacity:0.95; }

/* ---------- Fade-in for cards ---------- */
@keyframes fadeInUp {
    from {opacity:0; transform:translateY(14px);}
    to {opacity:1; transform:translateY(0);}
}
.food-card {
    background: white; border-radius: 20px; padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08); margin-bottom:15px;
    animation: fadeInUp 0.5s ease;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid rgba(76,175,80,0.12);
}
.food-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 16px 34px rgba(76,175,80,0.25);
}

.macro-badge {
    display:inline-block; padding:8px 14px; border-radius:12px;
    font-weight:600; font-size:14px; margin:5px;
    animation: pop 0.35s ease;
}
@keyframes pop { from{transform:scale(0.75); opacity:0;} to{transform:scale(1); opacity:1;} }

/* ---------- KPI cards ---------- */
.kpi-card {
    background: white; border-radius: 18px; padding: 18px;
    text-align:center; box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    animation: fadeInUp 0.5s ease; transition: transform 0.2s ease;
}
.kpi-card:hover { transform: translateY(-5px); }
.kpi-value { font-size:1.6rem; font-weight:800; margin:0; }
.kpi-label { font-size:0.85rem; color:#666; margin:0; }
.kpi-sub { font-size:0.75rem; color:#999; margin-top:4px; }

/* ---------- Animated progress bars ---------- */
.progress-wrap { background:#e8f5e9; border-radius:12px; height:18px; overflow:hidden; margin-bottom:10px; }
.progress-fill {
    height:100%; border-radius:12px;
    background: linear-gradient(90deg,#66bb6a,#2e7d32);
    animation: growBar 1s ease-out;
}
@keyframes growBar { from{width:0%;} }

/* ---------- Live camera pulse dot ---------- */
.pulse-dot {
    height:12px; width:12px; background:#e53935; border-radius:50%;
    display:inline-block; margin-right:8px; animation: pulse 1.4s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(229,57,53,0.6); }
    70% { box-shadow: 0 0 0 10px rgba(229,57,53,0); }
    100% { box-shadow: 0 0 0 0 rgba(229,57,53,0); }
}

/* ---------- Chat bubbles ---------- */
.chat-bubble-user {
    background:#dcf8c6; padding:10px 16px; border-radius:16px 16px 2px 16px;
    display:inline-block; max-width:80%; margin:6px 0; float:right; clear:both;
    animation: fadeInUp 0.3s ease;
}
.chat-bubble-bot {
    background:#f1f0f0; padding:10px 16px; border-radius:16px 16px 16px 2px;
    display:inline-block; max-width:80%; margin:6px 0; float:left; clear:both;
    animation: fadeInUp 0.3s ease;
}

/* Buttons glow on hover */
.stButton>button {
    transition: all 0.2s ease;
    border-radius: 12px !important;
}
.stButton>button:hover {
    box-shadow: 0 0 14px rgba(76,175,80,0.5);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# ⚙️ CONSTANTS
# =========================================================
CFG_IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FOOD_DB = {
    "salad": {"calories": 35, "protein": 2.5, "fat": 0.8, "carb": 6, "fiber": 2.5, "sugar": 2.2, "name": "Vegetable Salad"},
    "spaghetti": {"calories": 158, "protein": 5.8, "fat": 0.9, "carb": 30.9, "fiber": 1.8, "sugar": 0.6, "name": "Spicy Tomato Fusilli"},
    "sushi": {"calories": 130, "protein": 6, "fat": 1.5, "carb": 22, "fiber": 0.8, "sugar": 3, "name": "Salmon Bowl"},
    "pizza": {"calories": 266, "protein": 11, "fat": 10, "carb": 33, "fiber": 2.3, "sugar": 3.6, "name": "Pizza Slice"},
    "burger": {"calories": 295, "protein": 17, "fat": 14, "carb": 24, "fiber": 1.5, "sugar": 4, "name": "Chicken Burger"},
    "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carb": 14, "fiber": 2.4, "sugar": 10, "name": "Apple"},
    "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carb": 23, "fiber": 2.6, "sugar": 12, "name": "Banana"},
    "chicken breast": {"calories": 165, "protein": 31, "fat": 3.6, "carb": 0, "fiber": 0, "sugar": 0, "name": "Chicken Breast"},
    "rice": {"calories": 130, "protein": 2.4, "fat": 0.3, "carb": 28, "fiber": 0.4, "sugar": 0.05, "name": "Rice"},
    "salmon": {"calories": 208, "protein": 20, "fat": 13, "carb": 0, "fiber": 0, "sugar": 0, "name": "Salmon"},
    "yogurt": {"calories": 59, "protein": 3.5, "fat": 0.4, "carb": 5, "fiber": 0, "sugar": 3.2, "name": "Greek Yogurt"},
    "cake": {"calories": 371, "protein": 5, "fat": 16, "carb": 52, "fiber": 0.8, "sugar": 31, "name": "Chocolate Cake"},
}

# =========================================================
# 🗂️ SESSION STATE
# =========================================================
defaults = {
    "meals": [], "shopping_list": [], "chat_history": [],
    "user_profile": {"age": 25, "sex": "Male", "weight": 70, "height": 175, "activity": "Moderate", "goal": "Maintenance"},
    "daily_target": None,
    "last_label": None,
    "last_label_raw": "",
    "last_label_warn": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# 🧮 NUTRITION MATH HELPERS
# =========================================================
def calculate_bmr(w, h, age, sex):
    if sex == "Male":
        return 88.362 + (13.397 * w) + (4.799 * h) - (5.677 * age)
    return 447.593 + (9.247 * w) + (3.098 * h) - (4.330 * age)

def calculate_tdee(bmr, activity):
    mult = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
    return bmr * mult.get(activity, 1.55)

def calculate_goals(tdee, weight, goal):
    if goal == "Weight Loss": cal = tdee - 500
    elif goal == "Muscle Gain": cal = tdee + 300
    else: cal = tdee

    if goal == "Muscle Gain": p = weight * 2.2
    elif goal == "Weight Loss": p = weight * 2.0
    else: p = weight * 1.8

    f = (cal * 0.25) / 9
    c = max((cal - (p * 4 + f * 9)) / 4, 0)  # 🔧 fixed: clamp carbs at 0
    return {"calories": round(cal), "protein": round(p), "fat": round(f), "carb": round(c), "fiber": 25}

def get_today_meals():
    today = date.today().isoformat()
    return [m for m in st.session_state.meals if m.get("date") == today]

def get_totals():
    meals = get_today_meals()
    tot = {"calories": 0, "protein": 0, "fat": 0, "carb": 0, "fiber": 0}
    for m in meals:
        n = m["nutrition"]
        for k in tot:
            tot[k] += n.get(k, 0)
    return tot

# =========================================================
# 🧠 RGB-D MODEL (Nutrition5k checkpoint from Hugging Face)
# =========================================================
if HAS_TORCH:
    class NutritionNet(nn.Module):
        def __init__(self, backbone_name="convnext_small", in_chans=4, out_dim=5):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, in_chans=in_chans,
                num_classes=0, global_pool="avg"
            )
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 512),
                nn.GELU(), nn.Dropout(0.20), nn.Linear(512, out_dim)
            )

        def forward(self, x):
            return self.head(self.backbone(x))

    @st.cache_resource(show_spinner="🔽 Downloading Nutrition5k model (first run only)...")
    def load_nutrition_model():
        from huggingface_hub import hf_hub_download
        try:
            ckpt_path = hf_hub_download(
                repo_id="Anton-Atef/AI-nutrition-assistant",
                filename="best_nutrition_rgbd.pt",
                repo_type="model"
            )
        except Exception as e:
            st.warning(f"Could not download Nutrition5k checkpoint: {e}")
            return None, None, None
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
            y_std = np.array(ckpt["y_std"], dtype=np.float32)
            cols = ckpt.get("target_cols", ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"])
            model = NutritionNet(ckpt.get("backbone", "convnext_small"), in_chans=ckpt.get("in_chans", 4), out_dim=len(cols))
            model.load_state_dict(ckpt["model"])
            model.eval()
            return model, (y_mean, y_std), cols
        except Exception as e:
            st.warning(f"Could not load Nutrition5k checkpoint: {e}")
            return None, None, None
else:
    def load_nutrition_model():
        return None, None, None

@st.cache_resource
def load_food_classifier():
    try:
        from transformers import pipeline
        return pipeline("image-classification", model="nateraw/food", top_k=3)
    except Exception:
        return None

def predict_with_rgbd_model(pil_img, model_data):
    if model_data[0] is None:
        return None
    model, (y_mean, y_std), cols = model_data
    try:
        img = pil_img.resize((CFG_IMG_SIZE, CFG_IMG_SIZE))
        rgb = np.array(img).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        # 🔧 fixed: neutral mid-range depth (no physical sensor on phone/webcam)
        depth = np.full((CFG_IMG_SIZE, CFG_IMG_SIZE, 1), 0.5, dtype=np.float32)
        depth = (depth - 0.5) / 0.25

        img4 = np.concatenate([rgb, depth], axis=2)
        x = torch.from_numpy(img4).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            pred = model(x).numpy()[0]
        pred = pred * y_std + y_mean
        mapping = {c: v for c, v in zip(cols, pred)}
        return {
            "name": "AI Estimated Dish",
            "mass": float(max(mapping.get("total_mass", 250), 1)),
            "calories": float(max(mapping.get("total_calories", 200), 0)),
            "fat": float(max(mapping.get("total_fat", 10), 0)),
            "carb": float(max(mapping.get("total_carb", 18), 0)),
            "protein": float(max(mapping.get("total_protein", 12), 0)),
            "fiber": 2.0, "sugar": 3.0
        }
    except Exception as e:
        st.error(f"RGBD inference failed: {e}")
        return None

def estimate_from_food_db(label: str, grams: float):
    label = label.lower()
    best = None
    for k, v in FOOD_DB.items():
        if k in label:
            best = v
            break
    if not best:
        best = FOOD_DB["salad"]
    factor = grams / 100
    return {
        "name": best["name"],
        "calories": round(best["calories"] * factor, 1),
        "protein": round(best["protein"] * factor, 1),
        "fat": round(best["fat"] * factor, 1),
        "carb": round(best["carb"] * factor, 1),
        "fiber": round(best.get("fiber", 0) * factor, 1),
        "sugar": round(best.get("sugar", 0) * factor, 1),
        "mass": grams
    }

# =========================================================
# 🧾 NUTRITION LABEL OCR — (your requested style, kept as-is)
# =========================================================
EXPECTED_KEYS = [
    "product_name", "serving_size", "servings_per_container",
    "calories", "total_fat_g", "saturated_fat_g", "trans_fat_g",
    "cholesterol_mg", "sodium_mg", "total_carbohydrates_g",
    "dietary_fiber_g", "total_sugars_g", "added_sugars_g", "protein_g",
]

NUM_PAT = r"(\d+[.,]?\d*)"

OCR_PATTERNS = {
    "calories":              rf"calories(?!\s*from)\D{{0,15}}{NUM_PAT}",
    "total_fat_g":           rf"total\s*fat\D{{0,10}}{NUM_PAT}\s*g",
    "saturated_fat_g":       rf"sat(?:urated)?\.?\s*fat\D{{0,10}}{NUM_PAT}\s*g",
    "trans_fat_g":           rf"trans\s*fat\D{{0,10}}{NUM_PAT}\s*g",
    "cholesterol_mg":        rf"cholesterol\D{{0,10}}{NUM_PAT}\s*mg",
    "sodium_mg":             rf"sodium\D{{0,10}}{NUM_PAT}\s*mg",
    "total_carbohydrates_g": rf"total\s*carb(?:ohydrate)?s?\D{{0,10}}{NUM_PAT}\s*g",
    "dietary_fiber_g":       rf"(?:dietary\s*)?fib(?:er|re)\D{{0,10}}{NUM_PAT}\s*g",
    "total_sugars_g":        rf"total\s*sugars\D{{0,10}}{NUM_PAT}\s*g|(?<!added\s)sugars\D{{0,10}}{NUM_PAT}\s*g",
    "added_sugars_g":        rf"(?:incl\.?\s*)?added\s*sugars\D{{0,10}}{NUM_PAT}\s*g",
    "protein_g":             rf"protein\D{{0,10}}{NUM_PAT}\s*g",
}


def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Upscale + denoise + threshold to dramatically improve Tesseract accuracy."""
    if not HAS_CV2:
        return pil_img.convert("L")

    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale small images (helps OCR a lot)
    h, w = gray.shape
    if max(h, w) < 1500:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold (handles uneven lighting on photographed labels)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # Deskew (best-effort, safe fallback if it fails)
    try:
        coords = np.column_stack(np.where(thresh < 255))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h2, w2) = thresh.shape
            M = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w2, h2),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    return Image.fromarray(thresh)


def normalize_ocr_text(text: str) -> str:
    t = text.lower()
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    # common OCR confusions inside numbers
    t = re.sub(r"(?<=\d)o(?=\d|\s|g|mg)", "0", t)   # "1o0" -> "100"
    t = re.sub(r"(?<=\d)l(?=\s|g|mg)", "1", t)      # "10l" -> "101" (rare)
    t = t.replace(",", ".")                          # "1,200" thousand sep -> careful
    t = t.replace("º", "0").replace("O%", "0%")
    return t


def run_ocr_multi(pil_img: Image.Image) -> str:
    """Try a few PSM modes and keep whichever finds the most nutrition keywords."""
    if not TESS_OK:
        return ""

    processed = preprocess_for_ocr(pil_img)
    candidates = []
    for psm in [6, 4, 11, 3]:
        try:
            config = f"--psm {psm}"
            txt = pytesseract.image_to_string(processed, config=config)
            candidates.append(txt)
        except Exception:
            continue
    if not candidates:
        try:
            return pytesseract.image_to_string(pil_img)
        except Exception:
            return ""

    keywords = ["calories", "fat", "sodium", "carbohydrate", "protein", "sugar"]

    def score(t):
        low = t.lower()
        return sum(low.count(k) for k in keywords)

    return max(candidates, key=score)


def extract_nutrition_ocr(pil_img: Image.Image):
    if not TESS_OK:
        return None, "Tesseract is not available on this server (check packages.txt)."

    raw_text = run_ocr_multi(pil_img)
    norm = normalize_ocr_text(raw_text)

    result = {"product_name": None, "serving_size": None, "servings_per_container": None}

    for key, pattern in OCR_PATTERNS.items():
        m = re.search(pattern, norm)
        if m:
            val = next(g for g in m.groups() if g)  # first non-None captured group
            try:
                result[key] = float(val.replace(",", "."))
            except ValueError:
                result[key] = None
        else:
            result[key] = None

    sm = re.search(r"serving size\D{0,25}([\d./]+\s*\w*\s*\(?\d*\s*g?\)?)", norm)
    if sm:
        result["serving_size"] = sm.group(1).strip()

    spc = re.search(r"servings?\s*per\s*container\D{0,10}(\d+)", norm)
    if spc:
        result["servings_per_container"] = float(spc.group(1))

    result["raw_text"] = raw_text
    found_count = sum(1 for k in EXPECTED_KEYS if k not in ("product_name", "serving_size", "servings_per_container")
                       and result.get(k) is not None)

    if found_count == 0:
        return result, ("⚠️ Could not confidently detect any nutrition values. "
                         "Try a sharper, well-lit, straight-on photo of the label, "
                         "or fill values manually below.")
    return result, None

# =========================================================
# 🧭 AI ROUTER + CHATBOT
# =========================================================
@dataclass
class Route:
    intent: str
    tool: Optional[str]
    confidence: float
    arguments: Dict[str, Any] = field(default_factory=dict)

class AIRouter:
    def __init__(self):
        self.rules = {
            "daily_summary": [r"how many calories.*(left|remaining)", r"calories.*(left|remaining)", r"what did i eat today", r"daily.*summary"],
            "remaining_macros": [r"how much protein.*(left|remaining)", r"remaining.*(protein|carb|fat|macro)"],
            "portion_calculation": [r"how much.*can i eat", r"how many grams", r"what portion"],
            "food_logging": [r"i ate", r"log this", r"add this"],
            "recommendation": [r"what should i eat", r"recommend.*meal", r"what.*eat.*dinner"],
            "food_comparison": [r"compare", r"which.*better"],
            "user_targets": [r"my calorie target", r"my macros", r"what.*my.*target"],
            "food_history": [r"food history", r"what.*ate.*yesterday", r"show.*my meals"],
        }
        self.tool_map = {
            "daily_summary": "get_daily_summary", "remaining_macros": "get_remaining_macros",
            "portion_calculation": "calculate_food_portion", "food_logging": "log_food",
            "recommendation": "recommend_meals", "food_comparison": "compare_foods",
            "user_targets": "get_user_targets", "food_history": "get_food_history",
        }

    def route(self, message: str) -> Route:
        text = re.sub(r"\s+", " ", message.lower().strip())
        scores = {intent: sum(bool(re.search(p, text)) for p in pats) for intent, pats in self.rules.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return Route("general_nutrition_chat", None, 0.25)
        return Route(best, self.tool_map[best], min(0.95, 0.55 + scores[best] * 0.15))

router = AIRouter()

def generate_coach_response(user_msg):
    totals = get_totals()
    target = st.session_state.daily_target or calculate_goals(2000, 70, "Maintenance")
    remaining = {k: target[k] - totals.get(k, 0) for k in ["calories", "protein", "carb", "fat"]}
    route = router.route(user_msg)
    ctx = f"Today eaten: {totals}, Target: {target}, Remaining: {remaining}, Meals: {len(get_today_meals())}"

    if route.intent == "daily_summary":
        return (f"**Today's Summary** 📊\n\nYou have eaten {len(get_today_meals())} meals.\n\n"
                f"- Calories: {totals['calories']:.0f}/{target['calories']} (remaining {remaining['calories']:.0f})\n"
                f"- Protein: {totals['protein']:.0f}g / {target['protein']}g (remaining {remaining['protein']:.0f}g)\n"
                f"- Carbs: {totals['carb']:.0f}g / {target['carb']}g\n- Fat: {totals['fat']:.0f}g / {target['fat']}g\n\n"
                f"You're doing great! Keep it balanced.")

    if route.intent == "remaining_macros":
        return (f"You have **{remaining['calories']:.0f} kcal** left today.\n\n"
                f"- Protein: {remaining['protein']:.0f}g\n- Carbs: {remaining['carb']:.0f}g\n- Fat: {remaining['fat']:.0f}g\n\n"
                f"Want a meal suggestion to fill it?")

    if route.intent == "recommendation":
        if remaining['protein'] > 20:
            return (f"With {remaining['calories']:.0f} kcal left and {remaining['protein']:.0f}g protein needed, try:\n"
                    f"- **Grilled Chicken + Rice + Salad** (~400 kcal, 35g protein)\n"
                    f"- Greek Yogurt with fruits (~150 kcal)\n- Salmon Bowl (~350 kcal, 22g protein)")
        return f"Light option with {remaining['calories']:.0f} kcal left: **Vegetable Salad + Olive oil** or **Apple + Peanut Butter**."

    if route.intent == "portion_calculation":
        return (f"Tell me the food and I can calculate portion! You have {remaining['calories']:.0f} kcal remaining. "
                f"For example, chocolate cake is ~371 kcal/100g, so you could have about "
                f"{max(0, remaining['calories']/371*100):.0f}g.")

    if route.intent == "user_targets":
        return (f"Your daily targets ({st.session_state.user_profile['goal']}):\n"
                f"- {target['calories']} kcal\n- Protein {target['protein']}g\n- Carbs {target['carb']}g\n- Fat {target['fat']}g")

    if route.intent == "food_comparison":  # 🔧 now handled instead of falling through
        return ("Head to the **🛒 Smart Shopping** tab — pick two foods there and I'll compare "
                "calories, protein, fat, and sugar side by side, then tell you which is the better choice.")

    if route.intent == "food_history":  # 🔧 now handled
        meals = get_today_meals()
        if not meals:
            return "You haven't logged any meals today yet. Scan a meal in **📷 Live Scanner** to get started!"
        lines = "\n".join([f"- {m['time']} — {m['name']} ({m['nutrition']['calories']:.0f} kcal)" for m in meals])
        return f"**Today's food history:**\n\n{lines}"

    if route.intent == "food_logging":  # 🔧 now handled
        return ("I can't log meals from chat text yet — please use **📷 Live Scanner** to snap a photo, "
                "or **🏷️ Label Scanner** to scan a packaged product, and I'll add it to your log automatically.")

    # General LLM fallback (optional OpenAI key)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are an AI Nutrition Coach. Data: {ctx}. Be concise, friendly, no shaming."},
                    {"role": "user", "content": user_msg}
                ], max_tokens=300
            )
            return resp.choices[0].message.content
    except Exception:
        pass

    return (f"I'm your AI Nutrition Coach 🥗\n\n{ctx}\n\nYou asked: '{user_msg}'\n\n"
            f"Tips: Focus on whole foods, balance your remaining macros, and enjoy food without guilt. "
            f"What would you like to log or know about your meals?")

# =========================================================
# 👤 SIDEBAR — PROFILE & TARGETS
# =========================================================
with st.sidebar:
    st.title("👤 Your Profile")
    p = st.session_state.user_profile
    p["age"] = st.number_input("Age", 10, 90, p["age"])
    p["sex"] = st.selectbox("Sex", ["Male", "Female"], index=0 if p["sex"] == "Male" else 1)
    p["weight"] = st.number_input("Weight (kg)", 30.0, 200.0, float(p["weight"]))
    p["height"] = st.number_input("Height (cm)", 100.0, 230.0, float(p["height"]))
    p["activity"] = st.selectbox("Activity", ["Sedentary", "Light", "Moderate", "Active", "Very Active"], index=2)
    p["goal"] = st.selectbox("Goal", ["Weight Loss", "Maintenance", "Muscle Gain"],
                              index=["Weight Loss", "Maintenance", "Muscle Gain"].index(p["goal"]))

    bmr = calculate_bmr(p["weight"], p["height"], p["age"], p["sex"])
    tdee = calculate_tdee(bmr, p["activity"])
    target = calculate_goals(tdee, p["weight"], p["goal"])
    st.session_state.daily_target = target

    st.divider()
    st.markdown(f"""
    <div class="food-card" style="padding:14px;">
        <p class="kpi-label">🔥 TDEE</p>
        <p class="kpi-value" style="color:#2e7d32;">{tdee:.0f} kcal/day</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🎯 Daily Targets**")
    st.markdown(f"""
    <span class="macro-badge" style="background:#e8f5e9;">🔥 {target['calories']} kcal</span>
    <span class="macro-badge" style="background:#e3f2fd;">💪 {target['protein']}g P</span>
    <span class="macro-badge" style="background:#fff3e0;">🍞 {target['carb']}g C</span>
    <span class="macro-badge" style="background:#ffebee;">🥑 {target['fat']}g F</span>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Today's Log", use_container_width=True):
        st.session_state.meals = [m for m in st.session_state.meals if m["date"] != date.today().isoformat()]
        st.toast("Today's log cleared!", icon="🗑️")
        st.rerun()

# =========================================================
# 🎬 MAIN HEADER
# =========================================================
st.markdown("""
<div class="hero-header">
    <h1>🥗 Live AI Nutrition & Smart Shopping Assistant</h1>
    <p>Live camera • Nutrition5k model • OCR Label Scanner • AI Coach • Smart Shopping</p>
</div>
""", unsafe_allow_html=True)

# Load models (cached, lazy)
nutrition_model_data = load_nutrition_model()
food_classifier = load_food_classifier()

totals = get_totals()
remaining = {k: target[k] - totals.get(k, 0) for k in ["calories", "protein", "carb", "fat"]}

# =========================================================
# 📊 ANIMATED KPI ROW
# =========================================================
def kpi_card(icon, label, value, unit, sub, color):
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-label">{icon} {label}</p>
        <p class="kpi-value" style="color:{color};">{value:.0f}{unit}</p>
        <p class="kpi-sub">{sub}</p>
    </div>
    """, unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("🔥", "Calories Left", remaining["calories"], " kcal", f"{totals['calories']:.0f} eaten", "#e53935")
with k2: kpi_card("💪", "Protein Left", remaining["protein"], " g", f"{totals['protein']:.0f}g eaten", "#1e88e5")
with k3: kpi_card("🍞", "Carbs Left", remaining["carb"], " g", f"{totals['carb']:.0f}g eaten", "#fb8c00")
with k4: kpi_card("🥑", "Fat Left", remaining["fat"], " g", f"{totals['fat']:.0f}g eaten", "#43a047")

st.write("")  # spacing

tabs = st.tabs(["📷 Live Scanner", "🏷️ Label Scanner", "📊 Dashboard", "🤖 AI Coach", "🛒 Smart Shopping"])

# =========================================================
# 📷 TAB 1 — LIVE SCANNER
# =========================================================
with tabs[0]:
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown('<span class="pulse-dot"></span> **Live Camera Food Recognition**', unsafe_allow_html=True)
        st.caption("Point your camera at your meal — the AI estimates calories & macros instantly.")
        cam = st.camera_input("Take a picture of your meal")
        upl = st.file_uploader("Or upload food image", type=["jpg", "png", "jpeg"], key="food_up")
        img_file = cam if cam else upl
        portion = st.slider("Portion size (grams)", 50, 800, 350, step=10)

    with colB:
        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            st.image(pil_img, caption="Captured Meal", use_container_width=True)

            with st.spinner("🔍 Analyzing with Vision models..."):
                est = predict_with_rgbd_model(pil_img, nutrition_model_data)
                clf_label = "salad"
                clf_conf = 0.0
                if food_classifier:
                    try:
                        res = food_classifier(pil_img)
                        clf_label = res[0]["label"]
                        clf_conf = res[0]["score"]
                        st.info(f"Classifier: **{clf_label}** ({clf_conf:.1%})")
                    except Exception as e:
                        st.warning(f"Classifier error: {e}")

                if est is None:
                    est = estimate_from_food_db(clf_label, portion)
                else:
                    scale = portion / max(est["mass"], 1)
                    for k in ["calories", "protein", "fat", "carb"]:
                        est[k] = round(est[k] * scale, 1)
                    est["mass"] = portion
                    est["name"] = f"{clf_label.title()} ({est['name']})"

                st.markdown(f"""
                <div class="food-card">
                    <h3>{est['name']} — {est['mass']:.0f}g</h3>
                    <span class="macro-badge" style="background:#e8f5e9">🌾 Carbs: {est['carb']}g</span>
                    <span class="macro-badge" style="background:#fff3e0">💧 Fat: {est['fat']}g</span>
                    <span class="macro-badge" style="background:#ffebee">🍬 Sugar: {est.get('sugar',0)}g</span>
                    <h2 style="margin-top:15px">{est['calories']:.0f} kcal | P:{est['protein']}g C:{est['carb']}g F:{est['fat']}g</h2>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(data=[go.Pie(
                    labels=["Rice", "Salmon", "Cucumber", "Spinach", "Lettuce", "Sesame"],
                    values=[42.9, 28.6, 14.3, 4.4, 4.2, 5.7], hole=0.6
                )])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                                   transition_duration=500)
                st.plotly_chart(fig, use_container_width=True)

                if st.button("➕ Add to Today's Log", type="primary", use_container_width=True):
                    st.session_state.meals.append({
                        "id": len(st.session_state.meals),
                        "date": date.today().isoformat(),
                        "time": datetime.now().strftime("%H:%M"),
                        "name": est["name"],
                        "nutrition": est,
                    })
                    st.toast(f"Added {est['name']}!", icon="✅")
                    st.balloons()
        else:
            model_status = "Nutrition5k ConvNeXt ✅" if nutrition_model_data[0] else "FOOD-DB Demo Mode"
            st.info(f"📸 Start camera to scan your meal. Model: **{model_status}**")

# =========================================================
# 🏷️ TAB 2 — LABEL SCANNER (fixed nested-button bug)
# =========================================================
with tabs[1]:
    st.subheader("🏷️ Nutrition Facts Label Detection + OCR")

    if not TESS_OK:
        st.warning("⚠️ `pytesseract` is not available on this server. Add `pytesseract` to "
                    "`requirements.txt` and `tesseract-ocr` to `packages.txt`. You can still enter "
                    "values manually below after uploading a photo.")
    else:
        st.caption("OCR engine active: **pytesseract** (multi-PSM + auto-deskew preprocessing)")

    col1, col2 = st.columns(2)
    with col1:
        cam2 = st.camera_input("Photograph Nutrition Facts table", key="label_cam")
        upl2 = st.file_uploader("Upload label image", type=["jpg", "jpeg", "png"], key="label_up")
        img2_file = cam2 if cam2 else upl2
        st.caption("💡 Tip: flat label, good light, fill the frame, avoid glare/blur.")

    with col2:
        if img2_file:
            pil2 = Image.open(img2_file).convert("RGB")
            st.image(pil2, caption="Label Image", use_container_width=True)

            if st.button("🔍 Extract with OCR", type="primary", use_container_width=True):
                with st.spinner("Running OCR..."):
                    parsed, warn = extract_nutrition_ocr(pil2)
                st.session_state["last_label"] = parsed
                st.session_state["last_label_raw"] = parsed.get("raw_text", "") if parsed else ""
                st.session_state["last_label_warn"] = warn
        else:
            st.info("Take a photo of the Nutrition Facts table — OCR extracts calories, fat, carbs, sugar, etc.")

    # ---- Persistent results block (fixes the nested-button issue) ----
    if st.session_state.get("last_label") is not None:
        parsed = st.session_state["last_label"]
        warn = st.session_state.get("last_label_warn")

        st.markdown("---")
        if warn:
            st.warning(warn)

        st.markdown("#### 📋 Extracted Values — review & correct if needed")

        with st.expander("🔍 Debug: Raw OCR Text"):
            st.text(st.session_state.get("last_label_raw") or "No text captured.")

        numeric_fields = [k for k in EXPECTED_KEYS if k not in ("product_name", "serving_size", "servings_per_container")]
        edit_df = pd.DataFrame({"Nutrient": numeric_fields, "Value": [parsed.get(f) for f in numeric_fields]})
        edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed", key="label_editor")

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            prod_name = st.text_input("Product name for log", value=parsed.get("product_name") or "Packaged Product (OCR)")
        with cc2:
            serving_size = st.text_input("Serving size", value=parsed.get("serving_size") or "")
        with cc3:
            grams_basis = st.number_input("Values are per how many grams?", min_value=1, value=100, step=10)

        if st.button("➕ Add Label as Meal", use_container_width=True):
            vals = dict(zip(edited["Nutrient"], edited["Value"]))
            nutrition = {
                "calories": vals.get("calories") or 0,
                "fat": vals.get("total_fat_g") or 0,
                "protein": vals.get("protein_g") or 0,
                "carb": vals.get("total_carbohydrates_g") or 0,
                "fiber": vals.get("dietary_fiber_g") or 0,
                "sugar": vals.get("total_sugars_g") or 0,
                "mass": grams_basis,
                "name": prod_name
            }
            st.session_state.meals.append({
                "id": len(st.session_state.meals),
                "date": date.today().isoformat(),
                "time": datetime.now().strftime("%H:%M"),
                "name": nutrition["name"],
                "nutrition": nutrition
            })
            st.toast(f"Added {prod_name}!", icon="✅")
            st.balloons()
            st.session_state["last_label"] = None
            st.session_state["last_label_raw"] = ""
            st.session_state["last_label_warn"] = None
            st.rerun()

# =========================================================
# 📊 TAB 3 — DASHBOARD
# =========================================================
with tabs[2]:
    st.subheader("📊 Live Web Dashboard — Daily Progress")
    left, right = st.columns([1, 2])

    with left:
        totals = get_totals()
        fig = go.Figure(data=[go.Pie(
            labels=["Eaten", "Remaining"],
            values=[max(totals["calories"], 0), max(target["calories"] - totals["calories"], 0)],
            hole=0.7, marker_colors=["#66bb6a", "#e0e0e0"]
        )])
        fig.update_layout(title=f"Calories {totals['calories']:.0f}/{target['calories']}",
                           height=260, margin=dict(t=40, b=0, l=0, r=0), transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

        for macro, color in [("protein", "#1e88e5"), ("carb", "#fb8c00"), ("fat", "#43a047")]:
            pct = min(totals[macro] / target[macro], 1.0) if target[macro] > 0 else 0
            st.markdown(f"**{macro.title()}** — {totals[macro]:.0f}/{target[macro]}g")
            st.markdown(f"""
            <div class="progress-wrap">
                <div class="progress-fill" style="width:{pct*100}%; background: linear-gradient(90deg,{color}99,{color});"></div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        meals_today = get_today_meals()
        if meals_today:
            df = pd.DataFrame([{
                "Time": m["time"], "Food": m["name"], "kcal": m["nutrition"]["calories"],
                "P": m["nutrition"]["protein"], "C": m["nutrition"]["carb"], "F": m["nutrition"]["fat"]
            } for m in meals_today])
            st.dataframe(df, use_container_width=True)
            fig2 = px.bar(df, x="Food", y=["P", "C", "F"], barmode="group", title="Macros by Meal")
            fig2.update_layout(transition_duration=500)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No meals logged yet. Go to **📷 Live Scanner**!")

    st.divider()
    st.subheader("💡 Meal Suggestions based on Remaining Macros")
    rem = remaining
    suggestions = []
    if rem["protein"] > 20: suggestions.append("🔥 High Protein: Grilled Chicken 200g + Rice 100g (380 kcal, 42g P)")
    if rem["carb"] < 50: suggestions.append("🥑 Low Carb: Avocado + Eggs (250 kcal, 15g C)")
    if rem["calories"] > 400: suggestions.append("🍝 Balanced: Spicy Tomato Fusilli 250g (395 kcal)")
    if rem["calories"] < 150: suggestions.append("🍎 Light: Apple + Greek Yogurt (120 kcal)")
    for s in suggestions:
        st.markdown(f'<div class="food-card" style="padding:12px;">{s}</div>', unsafe_allow_html=True)

# =========================================================
# 🤖 TAB 4 — AI COACH
# =========================================================
with tabs[3]:
    st.subheader("🤖 AI Nutrition Chatbot — Talk to Your Data")
    st.caption("Router intents: daily_summary, remaining_macros, portion_calculation, recommendation, food_comparison, food_history...")

    st.markdown("#### 🎤 Voice Interaction")
    audio = st.audio_input("Ask with your voice")
    if audio:
        st.audio(audio)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio) as src:
                aud = r.record(src)
                text = r.recognize_google(aud)
                st.success(f"Transcribed: {text}")
                st.session_state.chat_history.append({"role": "user", "content": text})
                ans = generate_coach_response(text)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.warning(f"Voice transcription needs SpeechRecognition + internet: {e}. You can type instead.")

    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.chat_history:
            css_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
            st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="clear:both;"></div>', unsafe_allow_html=True)

    prompt = st.chat_input("e.g., How many calories do I have left? What should I eat for dinner?")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        ans = generate_coach_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()

# =========================================================
# 🛒 TAB 5 — SMART SHOPPING
# =========================================================
with tabs[4]:
    st.subheader("🛒 AI Smart Shopping — Compare & Better Alternatives")
    cA, cB = st.columns(2)

    with cA:
        st.markdown("**⚖️ Product Comparison**")
        prod1 = st.selectbox("Product A", list(FOOD_DB.keys()), index=0)
        prod2 = st.selectbox("Product B", list(FOOD_DB.keys()), index=1)
        if st.button("Compare", use_container_width=True):
            a = FOOD_DB[prod1]; b = FOOD_DB[prod2]
            comp_df = pd.DataFrame([a, b], index=[prod1, prod2])
            st.dataframe(comp_df, use_container_width=True)
            score_a = a["protein"] * 2 - a["sugar"] - a["fat"]
            score_b = b["protein"] * 2 - b["sugar"] - b["fat"]
            if score_a > score_b:
                st.success(f"✅ **{prod1}** is healthier (score {score_a:.1f} vs {score_b:.1f})")
                st.write(f"Swap tip: Choose {prod1} over {prod2} for more protein & less sugar.")
            else:
                st.success(f"✅ **{prod2}** is healthier (score {score_b:.1f} vs {score_a:.1f})")
                st.write(f"Swap tip: Choose {prod2} over {prod1} for more protein & less sugar.")

    with cB:
        st.markdown("**🧺 Shopping List**")
        new_item = st.text_input("Add food")
        if st.button("Add to List", use_container_width=True) and new_item:
            st.session_state.shopping_list.append(new_item)
            st.toast(f"Added '{new_item}' to list", icon="🛒")

        for i, item in enumerate(st.session_state.shopping_list):
            colx, coly = st.columns([4, 1])
            colx.checkbox(item, key=f"shop_{item}_{i}")  # 🔧 fixed stable key
            if coly.button("❌", key=f"del_{item}_{i}"):
                st.session_state.shopping_list.pop(i)
                st.rerun()

        st.markdown("**🤖 Better Alternatives AI**")
        st.markdown("""
        <div class="food-card" style="padding:14px;">
        🍰 Cake → Greek Yogurt + Berries <b>(save 200 kcal, +10g protein)</b><br><br>
        🥤 Soda → Sparkling Water + Lemon<br><br>
        🍚 White Rice → Quinoa <b>(more fiber & protein)</b>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("Built with Nutrition5k + Tesseract OCR + Rule-based/LLM Coach | © Live AI Nutrition Assistant")
