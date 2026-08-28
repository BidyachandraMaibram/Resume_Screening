"""
classifier.py — Two complementary ways to classify resumes by job fit:

1. score_resumes(): unsupervised, weighted scoring combining TF-IDF semantic
   similarity + required-skill match ratio + experience/education signals.
   Works immediately with no labeled training data — good default.

2. FitClassifier: supervised Logistic Regression classifier that learns from
   labeled examples (resume, job_description, label) where label is
   'fit' / 'not_fit'. Use this once you have historical hiring decisions
   to train on.
"""
from dataclasses import dataclass, field
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import (
    build_feature_row,
    tfidf_similarity_matrix,
    extract_skills,
)


@dataclass
class ScoredResume:
    filename: str
    fit_score: float          # 0-1 overall fit score
    similarity: float         # semantic similarity to JD
    skill_match_ratio: float  # fraction of required skills present
    matched_skills: set = field(default_factory=set)
    missing_skills: set = field(default_factory=set)
    years_experience: float = 0.0
    education_level: int = 0
    label: str = ""           # "Strong Fit" / "Possible Fit" / "Not a Fit"


def _bucket(score: float) -> str:
    if score >= 0.65:
        return "Strong Fit"
    elif score >= 0.4:
        return "Possible Fit"
    return "Not a Fit"


def score_resumes(job_description: str, resumes: dict, required_skills: set,
                   min_years: float = 0.0,
                   weights: dict = None) -> list:
    """
    Rank resumes against a job description without needing labeled training data.

    weights: relative importance of each signal, must sum to ~1.0
        {'similarity': 0.4, 'skills': 0.4, 'experience': 0.2}
    """
    weights = weights or {"similarity": 0.4, "skills": 0.4, "experience": 0.2}

    similarities = tfidf_similarity_matrix(job_description, resumes)
    results = []

    for fname, text in resumes.items():
        feats = build_feature_row(text, required_skills=required_skills)
        sim = similarities[fname]
        skill_ratio = feats.get("skill_match_ratio", 0.0)

        # experience score: capped at 1.0 once min_years is met (or 5 yrs if no min set)
        target_years = min_years if min_years > 0 else 5.0
        exp_score = min(feats["years_experience"] / target_years, 1.0) if target_years else 0.0

        fit_score = (
            weights["similarity"] * sim
            + weights["skills"] * skill_ratio
            + weights["experience"] * exp_score
        )

        results.append(ScoredResume(
            filename=fname,
            fit_score=round(fit_score, 4),
            similarity=round(sim, 4),
            skill_match_ratio=round(skill_ratio, 4),
            matched_skills=feats.get("matched_skills", set()),
            missing_skills=required_skills - feats.get("matched_skills", set()),
            years_experience=feats["years_experience"],
            education_level=feats["education_level"],
            label=_bucket(fit_score),
        ))

    results.sort(key=lambda r: r.fit_score, reverse=True)
    return results


class FitClassifier:
    """
    Supervised classifier: trains on (resume_text + job_description_text, label)
    pairs where label in {'fit', 'not_fit'}. Learns patterns beyond hand-tuned
    weights once you have real historical labels (e.g. past hiring outcomes).
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        self.is_trained = False

    @staticmethod
    def _combine(resume_text: str, job_description: str) -> str:
        # Concatenating JD + resume lets TF-IDF capture term overlap implicitly.
        return f"JOB: {job_description}\nRESUME: {resume_text}"

    def train(self, examples: list):
        """
        examples: list of dicts: {"resume": str, "job_description": str, "label": "fit"/"not_fit"}
        """
        X = [self._combine(ex["resume"], ex["job_description"]) for ex in examples]
        y = [ex["label"] for ex in examples]
        self.pipeline.fit(X, y)
        self.is_trained = True

    def predict(self, resume_text: str, job_description: str) -> dict:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet. Call .train() first.")
        combined = self._combine(resume_text, job_description)
        pred = self.pipeline.predict([combined])[0]
        proba = self.pipeline.predict_proba([combined])[0]
        classes = self.pipeline.named_steps["clf"].classes_
        proba_dict = dict(zip(classes, proba))
        return {"predicted_label": pred, "confidence": round(max(proba), 4), "probabilities": proba_dict}

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        self.pipeline = joblib.load(path)
        self.is_trained = True
