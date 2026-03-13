import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

app = FastAPI(title="ScreenGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "screen_guard_model.h5")
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


def analyze_image_damage(pil_image: Image.Image) -> float:
    """
    Heuristic image analysis to detect visual screen damage patterns:
    - Vertical/horizontal color stripe banding (LCD damage)
    - Abnormally high color saturation dispersion (dead pixels, backlight bleed)
    - Unusual color diversity (hardware failure patterns)
    Returns a damage score from 0.0 (healthy) to 1.0 (severely damaged).
    """
    # Work on a resized version for speed
    small = pil_image.resize((200, 200)).convert("RGB")
    arr = np.array(small, dtype=np.float32)  # (200, 200, 3)

    scores = []

    # ── 1. Vertical stripe detection ──────────────────────────────
    # Each column's mean color; adjacent-column color jumps = stripes
    col_means = arr.mean(axis=0)        # (200, 3)
    col_diffs = np.abs(np.diff(col_means, axis=0))  # (199, 3)
    vertical_stripe_score = col_diffs.mean() / 255.0
    scores.append(min(vertical_stripe_score * 8.0, 1.0))  # boosted multiplier

    # ── 2. Horizontal stripe detection ────────────────────────────
    row_means = arr.mean(axis=1)        # (200, 3)
    row_diffs = np.abs(np.diff(row_means, axis=0))
    horizontal_stripe_score = row_diffs.mean() / 255.0
    scores.append(min(horizontal_stripe_score * 8.0, 1.0))  # boosted multiplier

    # ── 3. High-saturation pixel ratio ───────────────────────────
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    max_ch = np.maximum(np.maximum(r, g), b)
    min_ch = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_ch > 10, (max_ch - min_ch) / (max_ch + 1e-5), 0.0)
    high_sat_ratio = float((sat > 0.6).mean())  # lowered threshold, more sensitive
    scores.append(min(high_sat_ratio * 2.5, 1.0))  # boosted multiplier

    # ── 4. Color diversity (damaged screens cycle many vivid hues) ─
    quantized = (arr / 32).astype(int)
    unique_colors = len(set(map(tuple, quantized.reshape(-1, 3))))
    color_diversity = unique_colors / (7 ** 3)   # normalise against max ~343
    scores.append(min(color_diversity * 2.0, 1.0))  # boosted multiplier

    # ── 5. Column consistency check ───────────────────────────────
    # Damaged stripe pattern: columns have very LOW within-column variance
    # but HIGH between-column variance → periodic pattern
    col_internal_std = arr.std(axis=0).mean()   # within each column
    col_external_std = col_means.std()           # across columns
    if col_internal_std > 0:
        stripe_ratio = col_external_std / (col_internal_std + 1e-5)
        scores.append(min(stripe_ratio / 2.0, 1.0))  # lower divisor = more sensitive

    # ── 6. Unnatural hue variety per row (LCD line failure) ────────
    # A damaged LCD often shows many distinct hues per horizontal band.
    # Compute per-row hue range and flag rows with extreme hue spread.
    hue_scores = []
    for row in arr:
        row_max = row.max(axis=0)
        row_min = row.min(axis=0)
        hue_range = float((row_max - row_min).mean()) / 255.0
        hue_scores.append(hue_range)
    avg_hue_spread = float(np.mean(hue_scores))
    scores.append(min(avg_hue_spread * 2.0, 1.0))

    # ── 7. Physical Crack Detection (Edge Density) ────────────────
    # Cracks appear as sharp, erratic high-frequency Edges.
    # We use a Laplacian-like kernel (difference of adjacent pixels).
    # Higher resolution is better for fine cracks.
    crack_res = pil_image.resize((300, 300)).convert("L")  # Grayscale for edge detection
    crack_arr = np.array(crack_res, dtype=np.float32)
    
    # Compute horizontal and vertical gradients
    dx = np.abs(np.diff(crack_arr, axis=1))[:, :-1]
    dy = np.abs(np.diff(crack_arr, axis=0))[:-1, :]
    mag = dx[:-1, :] + dy[:, :-1]
    
    # High gradients = potential cracks
    edge_density = float((mag > 35).mean()) # Threshold for "sharp" edges
    scores.append(min(edge_density * 15.0, 1.0)) # Cracks are thin, so density is naturally low

    heuristic_score = float(np.mean(scores))
    return heuristic_score


def get_damage_info(score: float):
    """
    Given a damage score (0=Normal, 1=Cracked/Damaged),
    return rich diagnostic information.
    """
    is_damaged = score > 0.45 # Slightly more sensitive threshold

    if is_damaged:
        damage_pct = round(max(0, (score - 0.45) / 0.55 * 100), 1)

        if damage_pct < 25:
            severity = "Minor"
            severity_color = "orange"
            urgency = "Low"
            recommendation = "Repair"
            action = (
                "A minor screen issue was detected. The screen may still be functional "
                "but shows early signs of damage. Consider getting it inspected to prevent further deterioration."
            )
            tips = [
                "Apply a tempered glass screen protector to stop any crack from spreading.",
                "Avoid pressing on the damaged area.",
                "Visit a certified repair centre within the next 2–4 weeks.",
                "Backup your data as a precaution.",
            ]
            repair_cost = "₹500 – ₹1,500"
            estimated_life = "6–12 months with care"

        elif damage_pct < 60:
            severity = "Moderate"
            severity_color = "amber"
            urgency = "Medium"
            recommendation = "Repair Soon"
            action = (
                "Moderate screen damage detected. Display issues such as color banding, "
                "flickering, or dead pixels are visible. Professional repair is strongly advised."
            )
            tips = [
                "Do not press on the screen — damage may worsen.",
                "Avoid exposing the device to heat or moisture.",
                "Schedule a professional repair within 1–2 weeks.",
                "Take a full device backup immediately.",
                "Minimise device usage until repaired.",
            ]
            repair_cost = "₹1,500 – ₹4,000"
            estimated_life = "2–4 months without repair"

        else:
            severity = "Severe"
            severity_color = "red"
            urgency = "High"
            recommendation = "Replace Screen"
            action = (
                "Severe screen damage detected. The display shows critical failure patterns "
                "(color lines, LCD bleed, or shattered pixels). Immediate screen replacement is required."
            )
            tips = [
                "Stop using the device immediately to avoid injury from glass or electrical hazard.",
                "Do NOT charge the device — damaged displays can cause short circuits.",
                "Backup data via PC if possible before the screen fails completely.",
                "Visit an authorised service centre for screen replacement.",
                "Request a full internal check — impact damage can affect other components.",
            ]
            repair_cost = "₹4,000 – ₹10,000+"
            estimated_life = "Replacement needed immediately"

    else:
        # Healthy screen — minor noise
        damage_pct = round((0.45 - score) * 2 * 5, 1)
        severity = "None"
        severity_color = "green"
        urgency = "None"
        recommendation = "No Action Needed"
        action = (
            "Your screen appears to be in excellent condition. "
            "No visible cracks, color distortion, or physical damage detected."
        )
        tips = [
            "Use a tempered glass screen protector to keep it safe.",
            "Use a quality phone case to guard against accidental drops.",
            "Clean with a microfiber cloth — avoid harsh chemicals.",
            "Avoid placing heavy objects on your device.",
        ]
        repair_cost = "N/A"
        estimated_life = "Normal lifespan expected"

    return {
        "damage_percentage": damage_pct,
        "severity": severity,
        "severity_color": severity_color,
        "urgency": urgency,
        "recommendation": recommendation,
        "action_message": action,
        "care_tips": tips,
        "estimated_repair_cost": repair_cost,
        "estimated_screen_life": estimated_life,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ── ML model prediction ───────────────────────────────────
        img_resized = pil_image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        ml_score = float(prediction[0][0])   # 0=Normal, 1=Cracked

        # ── Heuristic image analysis ──────────────────────────────
        heuristic_score = analyze_image_damage(pil_image)

        # ── Combine: trust heuristics when they are confident ──────
        # If heuristics are very confident about damage (like sharp cracks), 
        # let them drive the score, especially since the current model is a placeholder.
        if heuristic_score >= 0.6:
            # Very high heuristic confidence → nearly bypass ML
            final_score = 0.05 * ml_score + 0.95 * heuristic_score
        elif heuristic_score > 0.4:
            # Heuristics lean toward damage — heavy heuristic weight
            final_score = 0.2 * ml_score + 0.8 * heuristic_score
        elif heuristic_score < 0.2 and ml_score < 0.3:
            # Both agree it's healthy
            final_score = 0.4 * ml_score + 0.6 * heuristic_score
        else:
            # Uncertain — bias slightly towards heuristics for physical detection
            final_score = 0.3 * ml_score + 0.7 * heuristic_score

        label = "Cracked" if final_score > 0.45 else "Normal"
        confidence = final_score if final_score > 0.5 else (1 - final_score)

        damage_info = get_damage_info(final_score)

        return {
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_value": round(confidence * 100, 2),
            "final_score": round(final_score, 4),
            "status": "success",
            **damage_info,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
