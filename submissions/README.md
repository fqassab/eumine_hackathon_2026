# Submitting your predictions

## How to submit

1. **Fork** this repository (or, if you have write access, work on a branch)

2. Create a folder with your team name:
   ```
   submissions/YourTeamName/
   ```

3. Add your `predictions_test.json` file inside that folder.  
   The file must follow this format:
   ```json
   {
     "team_name": "YourTeamName",
     "model_id": "YourTeamName_v1",
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
   All 150 test `material_id` values must be present. The IDs correspond to the structures in `test_input_structures.zip` (Google Drive).

4. Open a **Pull Request** against the `main` branch.  
   CI will automatically check that your file is valid JSON with the correct fields.

5. The organizer will review and **merge** your PR.  
   Scores are updated on the leaderboard within a day of merging.

## Rules

- **Max 5 submissions** per team. Each PR = 1 submission.
- Only the **last merged** submission counts for the final ranking.
- Do not look at other teams' submissions before the deadline.

## What the score means

| Score | Meaning |
|-------|---------|
| 40 | Perfect (MAE → 0 for both properties) |
| 20 | Matches the baseline (Random Forest on MP data) |
| 0 | Much worse than baseline |

The performance score is 40% of your total Stage 1 score. The remaining 60% comes from jury review of your reports and code.

## questions?

Open a thread in [GitHub Discussions](../../../discussions).
