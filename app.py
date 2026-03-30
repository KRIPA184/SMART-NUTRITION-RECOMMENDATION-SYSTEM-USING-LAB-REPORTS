"""
Smart Nutrition Recommendation System – Flask Application
=========================================================
Main entry point. Orchestrates the 3-phase pipeline:
  Phase 1: Upload → OCR → Extraction → Validation
  Phase 2: ML Classification + Nutrient Prediction + SHAP
  Phase 3: Rule Engine → Health Summary → LLM Recommendation
"""

import os
import json
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for, flash, session, jsonify,
)
from werkzeug.utils import secure_filename

from utils.config import (
    SECRET_KEY, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    MEDICAL_DISCLAIMER,
)

# Directory for storing analysis results server-side (avoids cookie size limits)
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results_cache")
from services.ocr_service import process_upload
from services.extraction_service import extract_biomarkers, get_extraction_summary
from services.validation_service import validate_biomarkers, normalize_for_model
from services.rule_engine import compute_risk_score, generate_health_summary
from models.health_model import health_classifier
from models.nutrient_model import nutrient_predictor

# ─── App Initialization ──────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    GET  → Show upload form + patient info form.
    POST → Process the uploaded file through the full pipeline.
    """
    if request.method == "GET":
        return render_template("upload.html")

    # ── Validate file upload ──────────────────────────────────────
    if "lab_report" not in request.files:
        flash("No file selected. Please upload a lab report.", "danger")
        return redirect(url_for("upload"))

    file = request.files["lab_report"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Invalid file. Please upload a PDF, PNG, or JPG.", "danger")
        return redirect(url_for("upload"))

    # ── Collect patient info ──────────────────────────────────────
    # Gather allergies from checkboxes + free-text "other" field
    allergy_list = request.form.getlist("allergies")
    other_allergies = request.form.get("allergies_other", "").strip()
    if other_allergies:
        allergy_list += [a.strip() for a in other_allergies.split(",") if a.strip()]

    patient_info = {
        "name": request.form.get("patient_name", "").strip(),
        "age": request.form.get("age", "35"),
        "gender": request.form.get("gender", "other"),
        "activity_level": request.form.get("activity_level", "moderate"),
        "dietary_preference": request.form.get("dietary_preference", "non-vegetarian"),
        "cuisine_preference": request.form.get("cuisine_preference", "indian"),
        "allergies": allergy_list,
    }

    # Validate age
    try:
        patient_info["age"] = int(patient_info["age"])
        if not (1 <= patient_info["age"] <= 120):
            raise ValueError
    except (ValueError, TypeError):
        flash("Please enter a valid age (1-120).", "danger")
        return redirect(url_for("upload"))

    med_list = []

    # ── Save file ─────────────────────────────────────────────────
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(file_path)

    try:
        # ══════════════════════════════════════════════════════════
        #  PHASE 1 – Data Ingestion & Preprocessing
        # ══════════════════════════════════════════════════════════
        raw_text = process_upload(file_path)
        biomarkers = extract_biomarkers(raw_text)
        extraction_summary = get_extraction_summary(biomarkers)
        validation_result = validate_biomarkers(biomarkers)

        # ══════════════════════════════════════════════════════════
        #  PHASE 2 – AI/ML Model Core
        # ══════════════════════════════════════════════════════════
        ml_prediction = health_classifier.predict(biomarkers)
        shap_explanation = health_classifier.get_shap_explanation(biomarkers)
        nutrient_targets = nutrient_predictor.predict(biomarkers, patient_info)

        # ══════════════════════════════════════════════════════════
        #  PHASE 3 – Rule Engine & Health Summary
        # ══════════════════════════════════════════════════════════
        risk_result = compute_risk_score(validation_result["validated"])
        health_summary = generate_health_summary(
            biomarkers=biomarkers,
            validation_result=validation_result,
            risk_result=risk_result,
            patient_info=patient_info,
            nutrient_targets=nutrient_targets,
            shap_explanation=shap_explanation,
            medications=med_list,
        )

        # ══════════════════════════════════════════════════════════
        #  LLM Integration – Gemini (structured JSON only)
        # ══════════════════════════════════════════════════════════
        llm_result = {"success": False, "recommendation": None, "error": ""}
        try:
            from services.llm_service import get_nutrition_recommendation
            llm_result = get_nutrition_recommendation(health_summary)
        except Exception as e:
            llm_result["error"] = f"LLM service unavailable: {str(e)}"

        # ── Store results server-side (cookie has ~4KB limit) ────
        result_data = json.dumps({
            "patient_info": patient_info,
            "extraction_summary": extraction_summary,
            "biomarkers": biomarkers,
            "validation": validation_result,
            "ml_prediction": ml_prediction,
            "shap_explanation": shap_explanation,
            "nutrient_targets": nutrient_targets,
            "risk_result": risk_result,
            "health_summary": health_summary,
            "llm_result": llm_result,
            "disclaimer": MEDICAL_DISCLAIMER,
        }, default=str)

        result_id = uuid.uuid4().hex
        result_path = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result_data)

        session["result_id"] = result_id
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Error processing report: {str(e)}", "danger")
        return redirect(url_for("upload"))

    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/dashboard")
def dashboard():
    """Results dashboard – displays all analysis results."""
    result_id = session.get("result_id")
    if not result_id:
        flash("No results to display. Please upload a lab report first.", "info")
        return redirect(url_for("upload"))

    result_path = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    if not os.path.exists(result_path):
        flash("Results expired. Please upload a lab report again.", "info")
        return redirect(url_for("upload"))

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.loads(f.read())
    return render_template("dashboard.html", results=results)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    REST API endpoint for programmatic access.
    Accepts JSON with biomarker values and patient info directly.
    """
    data = request.get_json(force=True)

    biomarkers = data.get("biomarkers", {})
    patient_info = data.get("patient_info", {"age": 35, "gender": "other", "activity_level": "moderate"})
    medications = data.get("medications", [])

    # Phase 1 (skip OCR – data already structured)
    validation_result = validate_biomarkers(biomarkers)

    # Phase 2
    ml_prediction = health_classifier.predict(biomarkers)
    shap_explanation = health_classifier.get_shap_explanation(biomarkers)
    nutrient_targets = nutrient_predictor.predict(biomarkers, patient_info)

    # Phase 3
    risk_result = compute_risk_score(validation_result["validated"])
    health_summary = generate_health_summary(
        biomarkers=biomarkers,
        validation_result=validation_result,
        risk_result=risk_result,
        patient_info=patient_info,
        nutrient_targets=nutrient_targets,
        shap_explanation=shap_explanation,
        medications=medications,
    )

    # LLM
    llm_result = {"success": False, "recommendation": None}
    try:
        from services.llm_service import get_nutrition_recommendation
        llm_result = get_nutrition_recommendation(health_summary)
    except Exception as e:
        llm_result["error"] = str(e)

    return jsonify({
        "validation": validation_result,
        "ml_prediction": ml_prediction,
        "shap_explanation": shap_explanation,
        "nutrient_targets": nutrient_targets,
        "risk_result": risk_result,
        "llm_recommendation": llm_result,
        "disclaimer": MEDICAL_DISCLAIMER,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Nutrition AI Chatbot endpoint.
    Receives a user message + uses stored analysis results as context.
    Returns a plain-language AI response about nutrition, recipes, grocery lists, etc.
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Load patient context from stored results
    result_id = session.get("result_id")
    patient_context = {}
    if result_id:
        result_path = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                patient_context = json.loads(f.read())

    try:
        from services.llm_service import chat_with_nutritionist
        reply = chat_with_nutritionist(user_message, patient_context)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Sorry, I'm having trouble right now. Please try again. ({str(e)})"}), 500


# ─── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pre-train models on startup so first request is fast
    print("[*] Training health classification model...")
    health_classifier.train()
    print("[*] Training nutrient prediction model...")
    nutrient_predictor.train()
    print("[OK] Models ready. Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
