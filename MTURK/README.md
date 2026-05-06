# RippleBench MTurk Human Evaluation

Four experiments validating distinct stages of the RippleBench pipeline.  
See `mturk_plan.md` for the full specification.

## Quick Start

### 1. Generate HIT input CSVs

```bash
# All experiments (from repo root):
python MTURK/exp1_ranking/generate_hits.py
python MTURK/exp2_question_quality/generate_hits.py
python MTURK/exp3_article_topic/generate_hits.py --use-facts   # offline mode (uses extracted facts)
python MTURK/exp3_article_topic/generate_hits.py               # online mode (fetches from Wikipedia API)
python MTURK/exp4_topic_extraction/generate_hits.py
```

Each produces a CSV in its experiment directory (e.g., `exp1_ranking/exp1_hits.csv`).

### 2. Upload to MTurk

For each experiment:

1. **Go to** [MTurk Requester](https://requester.mturk.com/) → Create → New Project
2. **Select** "Survey" template
3. **Configure the project:**
   - Title: see table below
   - Reward: $0.05
   - Workers per HIT: 3
   - Time allotment: 5 minutes
   - Auto-approve: 48 hours
   - Qualifications: Masters
4. **Design Layout:** paste the contents of `template.html` for that experiment
5. **Publish:** upload the experiment's CSV as the HIT input file

| Exp | Title | CSV | Template |
|-----|-------|-----|----------|
| 1 | Topic Relatedness Judgment | `exp1_ranking/exp1_hits.csv` | `exp1_ranking/template.html` |
| 2 | Multiple Choice Question (Biology) | `exp2_question_quality/exp2_hits.csv` | `exp2_question_quality/template.html` |
| 3 | Article Topic Matching | `exp3_article_topic/exp3_hits.csv` | `exp3_article_topic/template.html` |
| 4 | Question Topic Matching | `exp4_topic_extraction/exp4_hits.csv` | `exp4_topic_extraction/template.html` |

### 3. Using the MTurk CLI (alternative)

If you prefer the AWS CLI / boto3:

```bash
pip install boto3
```

Create HITs programmatically using the CSV as input and the HTML template as the question XML.  
The template files are already in MTurk `HTMLQuestion` XML format.

## Experiment Summary

| # | Name | Pipeline Stage | HITs | Design |
|---|------|---------------|------|--------|
| 1 | Ranking Validation | base-topic → ranked neighbors | 250 | Pairwise comparison: which topic is more related? |
| 2 | Question Quality | facts → MCQ | 250 | Answer MCQ with/without facts (between-subject) |
| 3 | Article → Topic | topic → wiki article | 250 | Match article excerpt to correct topic (4-AFC) |
| 4 | Topic Extraction | WMDP question → topic | 250 | Match WMDP question to extracted topic (4-AFC) |

## Budget

- 4 experiments × 250 HITs × 3 workers × $0.05 = $150
- MTurk fees (20% Masters): $30
- **Total: ~$180**

## Notes

- Exp 3 has two modes: `--use-facts` uses the pre-extracted facts from the dataset as the article text (no internet needed); without the flag, it fetches full Wikipedia articles via the API (slower, requires `pip install wikipedia`).
- All generators use `random.seed(42)` for reproducibility.
- All templates record `time_spent_seconds` for quality filtering.
