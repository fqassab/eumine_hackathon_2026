# Leaderboard — EuMINe DataBridge Hackathon 2026

*Updated: 2026-05-07 17:22 UTC*

Score = `performance_score` (0–40 pts). Baseline score = 20 pts. ✓ = beats baseline.

| Rank | Team | Score /40 | MAE EF (eV/atom) | MAE BG (eV) | Missing |
|------|------|----------:|----------------:|------------:|--------:|
| 1 | OrganizerTest | **27.28** ✓ | 0.1547 | 0.4122 | 0 |

---

Performance score formula (per property): `score_p = 10 + 10 × (baseline_MAE − MAE) / (baseline_MAE − 0.01)` when beating baseline, else `score_p = max(0, 10 × (1 − (MAE − baseline) / baseline))`.

Baseline MAE: formation energy = 0.2378 eV/atom, band gap = 0.6414 eV.
