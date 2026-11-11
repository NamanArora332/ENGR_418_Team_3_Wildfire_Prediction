# Add annotated ENGR 418 wildfire prediction models

Closes #<replace with the actual new GitHub Issue number>

## What this adds
- Full line-commented `train_wildfire_model.py`, with links in comments to ENGR 418 Lectures 3, 4, 6, 7 and 8.
- Lecture 7 multinomial-logistic baseline, a simple four-leaf tree, a majority baseline, and an advanced XGBoost model.
- Conservative predictor set by default; separate **retrospective** feature policy for possible 2020-vintage inputs.
- Chronological 2020 training/validation/test split, rare-large-fire F2 threshold selection, reproducible metrics and figures.
- `requirements.txt`, `MODEL_README.md`, `feature_availability_audit.csv`, and three safety tests.

## Verification
- [x] Tested on actual `2020_FPA_FOD_cons.csv` (73,221 raw rows; 73,213 unique fire IDs).
- [x] `python -m unittest discover -s tests -v` passes 3 tests.
- [x] Actual benchmark CSVs and plots included for team review.
- [ ] A teammate has read the feature vintage audit and confirmed intended prediction-time assumptions.
- [ ] A teammate has independently reproduced the result table.
- [ ] Team has updated the main README as appropriate.
- [ ] Team has recorded the use of ChatGPT in the individual GenAI declaration.

## Limits
This dataset is 2020 only. It is not the 1992–2020 multi-year reproduction planned originally. The four-leaf baseline is a proxy, not Coffield's exact model. The optional retrospective feature mode is not verified for first-report deployment. This is course research, not operational wildfire guidance.
