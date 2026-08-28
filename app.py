"""
app.py — Web UI for the Resume Screening System

Run:
    pip install -r requirements.txt
    python app.py
"""
import os

from flask import Flask, jsonify, render_template, request

from classifier import score_resumes
from features import DEFAULT_SKILLS
from parser import clean_text, extract_text_from_bytes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
EDUCATION_LABELS = {0: "Not detected", 1: "Associate / Diploma", 2: "Bachelor's", 3: "Master's / MBA", 4: "PhD / Doctorate"}


def _serialize_result(result):
    return {
        "filename": result.filename,
        "fit_score": result.fit_score,
        "similarity": result.similarity,
        "skill_match_ratio": result.skill_match_ratio,
        "matched_skills": sorted(result.matched_skills),
        "missing_skills": sorted(result.missing_skills),
        "years_experience": result.years_experience,
        "education_level": result.education_level,
        "education_label": EDUCATION_LABELS.get(result.education_level, "Unknown"),
        "label": result.label,
    }


@app.route("/")
def index():
    sample_job = ""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "job_description.txt")
    if os.path.isfile(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            sample_job = f.read()
    return render_template("index.html", default_skills=sorted(DEFAULT_SKILLS.keys()), sample_job=sample_job)


@app.route("/api/extract-text", methods=["POST"])
def extract_text_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded."}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    try:
        text = clean_text(extract_text_from_bytes(f.read(), f.filename))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"text": text})


@app.route("/api/screen", methods=["POST"])
def screen():
    job_text = request.form.get("job_description", "").strip()
    if not job_text:
        return jsonify({"error": "Job description is required."}), 400

    files = request.files.getlist("resumes")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "Upload at least one resume file."}), 400

    try:
        min_years = float(request.form.get("min_years", "0") or "0")
    except ValueError:
        return jsonify({"error": "Minimum years must be a number."}), 400

    skills_raw = request.form.get("skills", "")
    required_skills = {s.strip().lower() for s in skills_raw.split(",") if s.strip()}

    resumes = {}
    errors = []
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{f.filename}: unsupported file type")
            continue
        try:
            raw = extract_text_from_bytes(f.read(), f.filename)
            resumes[f.filename] = clean_text(raw)
        except Exception as exc:
            errors.append(f"{f.filename}: {exc}")

    if not resumes:
        msg = "Could not read any resume files."
        if errors:
            msg += " " + "; ".join(errors)
        return jsonify({"error": msg}), 400

    results = score_resumes(
        job_description=clean_text(job_text),
        resumes=resumes,
        required_skills=required_skills,
        min_years=min_years,
    )

    top = request.form.get("top")
    if top:
        try:
            results = results[: int(top)]
        except ValueError:
            pass

    return jsonify({
        "results": [_serialize_result(r) for r in results],
        "warnings": errors,
        "total_screened": len(resumes),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
