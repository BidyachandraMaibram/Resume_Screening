# Resume Screening System

Classifies resumes by job fit using NLP feature extraction + classification algorithms.

## What it does

1. **Parses** resumes from `.txt`, `.pdf`, or `.docx` files.
2. **Extracts NLP features**: skills (via taxonomy matching), years of experience
   (regex), and education level.
3. **Scores fit** two ways:
   - `score_resumes()` — unsupervised weighted scoring (TF-IDF semantic similarity
     to the job description + skill match ratio + experience). Works immediately,
     no training data needed. This is the default / recommended path.
   - `FitClassifier` — supervised Logistic Regression that learns from labeled
     historical examples (`fit` / `not_fit`). Use once you have real past hiring
     outcomes to train on.
4. **Ranks and labels** candidates as `Strong Fit`, `Possible Fit`, or `Not a Fit`.

## Project structure

```
resume_screening/
├── app.py             # Web UI (Flask)
├── main.py           # CLI entry point
├── parser.py          # File → text extraction (pdf/docx/txt)
├── features.py         # Skill/experience/education extraction, TF-IDF similarity
├── classifier.py        # Scoring logic + supervised FitClassifier
├── templates/         # HTML templates
├── static/css/        # Stylesheets
├── requirements.txt
├── data/
│   ├── job_description.txt
│   └── sample_resumes/   # 4 example resumes with varying fit
└── README.md
```

## Web UI

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser. You can:

- Paste or upload a job description
- Drag-and-drop multiple resumes (.txt, .pdf, .docx)
- Set required skills, minimum experience, and optional top-N limit
- View ranked results with fit scores, matched/missing skills, and labels

Click **Load sample job description** to try the demo with the included ML Engineer role, then upload the files from `data/sample_resumes/`.

## CLI quick start

```bash
pip install scikit-learn nltk PyPDF2 python-docx joblib

python main.py \
  --job data/job_description.txt \
  --resumes data/sample_resumes \
  --skills "python,nlp,machine learning,cloud,deep learning" \
  --min-years 3
```

Add `--json` for machine-readable output, `--top 5` to limit results.

## Example output

```
Candidate             Fit%    Sim%    Skill%  Yrs     Label
--------------------------------------------------------------------
resume_alice.txt      72.5    31.3    100.0   5.0     Strong Fit
resume_david.txt      48.7    11.8    60.0    6.0     Possible Fit
resume_carla.txt      41.2    23.1    60.0    2.0     Possible Fit
resume_bob.txt        34.3    5.7     40.0    4.0     Not a Fit
```

## Using the supervised classifier

Once you have labeled examples (e.g. exported from past ATS decisions):

```python
from classifier import FitClassifier

examples = [
    {"resume": "...", "job_description": "...", "label": "fit"},
    {"resume": "...", "job_description": "...", "label": "not_fit"},
    # ... more labeled examples, ideally 50+ per class
]

clf = FitClassifier()
clf.train(examples)
clf.save("fit_model.joblib")   # persist for reuse

result = clf.predict(resume_text, job_description_text)
# {'predicted_label': 'fit', 'confidence': 0.83, 'probabilities': {...}}
```

## Customizing the skills taxonomy

Edit `DEFAULT_SKILLS` in `features.py` to match your domain — it's a dict of
`canonical_skill: [synonyms]`. This is what both the CLI `--skills` flag and the
unsupervised scorer key off of.

## Known limitations (important — read before production use)

- **Keyword matching has no negation detection.** A resume that says *"no NLP
  experience"* will register a match on "NLP" — the sample data intentionally
  includes this case (`resume_david.txt`) so you can see it happen. For
  production use, either add negation-window filtering (e.g. spaCy dependency
  parsing to detect negation scope) or rely more heavily on the TF-IDF
  similarity signal and less on raw keyword presence.
- **TF-IDF similarity is lexical, not semantic.** It won't recognize that
  "led a squad of engineers" means the same thing as "leadership" unless the
  words overlap. For stronger semantic matching, swap TF-IDF for sentence
  embeddings (e.g. `sentence-transformers`) and cosine similarity on those
  vectors instead.
- **The supervised classifier needs real labeled data to be trustworthy.**
  The demo trains on 6 toy examples purely to show the mechanics — a
  production model needs a much larger, representative, and bias-audited
  labeled dataset before it should influence real hiring decisions.
- **Legal/compliance**: automated resume screening is subject to regulations
  in many jurisdictions (e.g. NYC Local Law 144, EU AI Act provisions on
  employment). Any production deployment should include human review, bias
  audits across protected classes, and a documented appeals process.

## Extending this system

- Swap TF-IDF for embeddings: `sentence-transformers` + `cosine_similarity`
  for better semantic matching.
- Add a resume section parser (split into Experience / Education / Skills
  blocks) for more precise feature extraction per section.
- Log predictions + human overrides over time to build a real labeled dataset
  for the supervised classifier.
