# EuMINe DataBridge Hackathon 2026

**Predict materials properties from heterogeneous, multi-source data — and make your model interoperable.**

- **Stage 1 (remote):** May 11 – June 22, 2026
- **Stage 2 (in-person):** EuMINe General Meeting, Cluj-Napoca, July 2026
- **Submissions:** open a PR to [`submissions/YourTeam/predictions_test.json`](submissions/) — score posted automatically
- **Website:** https://www.eumine-cost.eu/news/eumine-hackathon-2026/
- **Q&A:** [GitHub Discussions](../../discussions)

---

## The Challenge

Given inorganic crystal structures (CIF), predict:

| Property | Unit |
|----------|------|
| Formation energy per atom | eV/atom |
| Electronic band gap | eV |

Use **at least two** public databases: Materials Project, JARVIS-DFT, AFLOW, NOMAD, or OQMD.  
Every submission must also implement the **MatFed API v1** (see below) so models can be combined during Stage 2.

---

## Scoring (100 points)

| Criterion | Points | How |
|-----------|--------|-----|
| Predictive performance | 40 | Automated MAE on held-out test set. Baseline = 20 pts. |
| Data integration quality | 25 | Jury review of data integration report |
| Federation readiness | 20 | Automated (MatFed tests pass = 15 pts) + jury (5 pts) |
| Reproducibility | 15 | Jury: can the code be run? |

Beat the baseline on at least one property to qualify for Stage 2.

---

## Repository Contents

### `matfed-api-template/`

The interface all submissions must implement. Start here.

```bash
cd matfed-api-template
pip install -r requirements.txt
pytest tests/test_interface.py -v   # must pass before submitting
```

See [`matfed-api-template/README.md`](matfed-api-template/README.md) for the full interface contract and a working example implementation.

### `baseline/`

Reference Random Forest model (MAGPIE + Materials Project data).

```bash
cd baseline
pip install -r requirements.txt
python run_baseline.py   # requires datasets downloaded from Google Drive
```

### `report_template/`

LaTeX templates for the two mandatory reports (max 4 pages each).  
See [`report_template/README.txt`](report_template/README.txt) for compilation and naming conventions.

---

## Datasets

Download from Google Drive: **https://drive.google.com/drive/folders/1CAF8_rymTdr-2PM9z-xy2RnSYJKi8XS2?usp=drive_link** *(link added when files are uploaded)*

| File | Description |
|------|-------------|
| `bridge_dataset_train.csv` | 700 materials with formation energy + band gap labels |
| `bridge_dataset_train_structures.zip` | CIF files for training set |
| `bridge_dataset_val.csv` | 150 materials with labels |
| `bridge_dataset_val_structures.zip` | CIF files for validation set |
| `test_input_structures.zip` | 150 CIF files — **no labels** (for final submission) |
| `data_card.pdf` | Dataset provenance, statistics, known caveats |
| `database_access_guide.pdf` | How to query MP, JARVIS, AFLOW, NOMAD, OQMD |

Test labels are never released — used only by the automated scoring bot.

---

## Submission Format

Add `submissions/YourTeam/predictions_test.json` and open a PR (max 5 PRs per team):

```json
{
  "team_name": "YourTeam",
  "model_id": "YourTeam_v1",
  "matfed_api_version": "1.0",
  "predictions": [
    {
      "material_id": "mp-XXXX",
      "formation_energy_per_atom": -1.234,
      "band_gap": 2.15
    }
  ]
}
```

All 150 test material IDs must be present. Within minutes of opening the PR, the scoring bot posts your performance score as a comment. The organizer merges the PR to record the submission on the [leaderboard](LEADERBOARD.md).

See [`submissions/README.md`](submissions/README.md) for full instructions.

---

## Team Rules

- 2–5 members from **at least 2 different institutions**
- Public GitHub repo, or private repo with organizer access (instructions in welcome email)
- Pre-trained models allowed — declare them in your technical report
- One person per team maximum

---

## Stage 2 (July 2026, Cluj-Napoca)

Top 8 teams are invited in person. EuMINe covers travel (up to €300/person) and accommodation for up to 3 representatives per team.

**Federation Sprint:** teams are grouped (3–4 teams/group) and must combine their models into a federated ensemble, evaluated on new out-of-distribution materials. The MatFed API makes this possible.

---

## Contact

- **Q&A:** [GitHub Discussions](../../discussions) — answers within 48h on working days
- **Email:** euminecost@gmail.com
