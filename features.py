"""
features.py — NLP feature engineering for resumes:
  - skill extraction via keyword/phrase matching (with simple synonym handling)
  - years-of-experience extraction via regex
  - education level detection
  - TF-IDF vectorization for semantic similarity
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# A reasonably broad default skills taxonomy. Extend/replace per role.
DEFAULT_SKILLS = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js", "typescript"],
    "sql": ["sql", "mysql", "postgresql", "postgres"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "pytorch", "tensorflow", "keras"],
    "nlp": ["nlp", "natural language processing", "spacy", "nltk", "transformers"],
    "data analysis": ["data analysis", "pandas", "numpy", "data analytics"],
    "cloud": ["aws", "azure", "gcp", "google cloud", "cloud computing"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "react": ["react", "react.js", "reactjs"],
    "communication": ["communication", "presentation", "public speaking"],
    "leadership": ["leadership", "team lead", "managed a team", "mentored"],
    "project management": ["project management", "agile", "scrum", "jira"],
}

EDUCATION_LEVELS = {
    "phd": 4, "doctorate": 4,
    "master": 3, "m.s.": 3, "msc": 3, "mba": 3,
    "bachelor": 2, "b.s.": 2, "bsc": 2, "b.tech": 2, "be ": 2,
    "associate": 1, "diploma": 1,
}


def extract_skills(text: str, taxonomy: dict = None) -> set:
    """Return the set of canonical skills found in text based on a taxonomy."""
    taxonomy = taxonomy or DEFAULT_SKILLS
    text_l = text.lower()
    found = set()
    for canonical, synonyms in taxonomy.items():
        for syn in synonyms:
            # word-boundary-ish match, tolerant of punctuation like "c++"
            pattern = re.escape(syn)
            if re.search(pattern, text_l):
                found.add(canonical)
                break
    return found


def extract_years_experience(text: str) -> float:
    """
    Heuristic extraction of total years of experience.
    Looks for patterns like '5 years of experience', '3+ years'.
    Falls back to 0 if nothing found.
    """
    text_l = text.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", text_l)
    years = [float(m) for m in matches]
    return max(years) if years else 0.0


def extract_education_level(text: str) -> int:
    """Return highest education level found (0 = none detected, 4 = PhD)."""
    text_l = text.lower()
    level = 0
    for keyword, score in EDUCATION_LEVELS.items():
        if keyword in text_l:
            level = max(level, score)
    return level


def build_feature_row(text: str, required_skills: set = None) -> dict:
    """Build a structured feature dict for one resume."""
    skills = extract_skills(text)
    row = {
        "skills": skills,
        "num_skills": len(skills),
        "years_experience": extract_years_experience(text),
        "education_level": extract_education_level(text),
    }
    if required_skills:
        matched = skills & required_skills
        row["matched_skills"] = matched
        row["skill_match_ratio"] = len(matched) / max(len(required_skills), 1)
    return row


def tfidf_similarity_matrix(job_description: str, resumes: dict):
    """
    Compute cosine similarity between a job description and each resume
    using TF-IDF vectors. Returns {filename: similarity_score (0-1)}.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    filenames = list(resumes.keys())
    corpus = [job_description] + [resumes[f] for f in filenames]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    tfidf = vectorizer.fit_transform(corpus)

    job_vec = tfidf[0:1]
    resume_vecs = tfidf[1:]
    sims = cosine_similarity(job_vec, resume_vecs)[0]

    return {fname: float(score) for fname, score in zip(filenames, sims)}
