"""
main.py — Resume Screening System CLI

Usage:
    python main.py --job data/job_description.txt --resumes data/sample_resumes --skills "python,nlp,machine learning,cloud"

Outputs a ranked table of candidates by job fit score, with matched/missing
skills, experience, and a Strong/Possible/Not-a-Fit label.
"""
import argparse
import json
import sys

from parser import load_resumes_from_dir, extract_text, clean_text
from classifier import score_resumes


def parse_args():
    p = argparse.ArgumentParser(description="Classify resumes by job fit using NLP.")
    p.add_argument("--job", required=True, help="Path to job description (.txt/.pdf/.docx)")
    p.add_argument("--resumes", required=True, help="Directory containing resume files")
    p.add_argument("--skills", default="",
                   help="Comma-separated required skills (canonical names from features.DEFAULT_SKILLS)")
    p.add_argument("--min-years", type=float, default=0.0, help="Minimum years of experience desired")
    p.add_argument("--json", action="store_true", help="Output results as JSON instead of a table")
    p.add_argument("--top", type=int, default=None, help="Only show top N candidates")
    return p.parse_args()


def print_table(results, top=None):
    rows = results[:top] if top else results
    col_widths = [22, 8, 8, 8, 8, 14]
    header = ["Candidate", "Fit%", "Sim%", "Skill%", "Yrs", "Label"]
    print("".join(h.ljust(w) for h, w in zip(header, col_widths)))
    print("-" * sum(col_widths))
    for r in rows:
        print("".join([
            r.filename.ljust(col_widths[0]),
            f"{r.fit_score*100:.1f}".ljust(col_widths[1]),
            f"{r.similarity*100:.1f}".ljust(col_widths[2]),
            f"{r.skill_match_ratio*100:.1f}".ljust(col_widths[3]),
            f"{r.years_experience:.1f}".ljust(col_widths[4]),
            r.label.ljust(col_widths[5]),
        ]))
    print()
    for r in rows:
        print(f"{r.filename}:")
        print(f"  Matched skills : {', '.join(sorted(r.matched_skills)) or '(none)'}")
        print(f"  Missing skills : {', '.join(sorted(r.missing_skills)) or '(none)'}")
        print(f"  Education level: {r.education_level} (0=none .. 4=PhD)")
        print()


def main():
    args = parse_args()

    job_text = clean_text(extract_text(args.job))
    resumes = load_resumes_from_dir(args.resumes)

    if not resumes:
        print(f"No resumes found in {args.resumes}", file=sys.stderr)
        sys.exit(1)

    required_skills = set(s.strip().lower() for s in args.skills.split(",") if s.strip())

    results = score_resumes(
        job_description=job_text,
        resumes=resumes,
        required_skills=required_skills,
        min_years=args.min_years,
    )

    if args.json:
        out = [{
            "filename": r.filename,
            "fit_score": r.fit_score,
            "similarity": r.similarity,
            "skill_match_ratio": r.skill_match_ratio,
            "matched_skills": sorted(r.matched_skills),
            "missing_skills": sorted(r.missing_skills),
            "years_experience": r.years_experience,
            "education_level": r.education_level,
            "label": r.label,
        } for r in (results[:args.top] if args.top else results)]
        print(json.dumps(out, indent=2))
    else:
        print_table(results, top=args.top)


if __name__ == "__main__":
    main()
