# Contributing — ENGR 418 Team 3

This course project should show meaningful, reviewable contributions from every member.

## Workflow

1. Pick or create a GitHub **Issue** describing the goal, proposed approach, verification steps and owner.
2. Create a branch from the up-to-date `main`, named `feature/<topic>` or `fix/<topic>`.
3. Make a focused change, document any dataset assumptions, run the tests, and commit with a descriptive message.
4. Create a **Pull Request** into `main`, referring to the Issue (`Closes #N` only when all acceptance criteria are satisfied).
5. Ask a *different* team member to review, respond to comments, then merge after approval.
6. Repeat for each member; keep discussions, meeting feedback and evidence in Issues / PRs.

### Local verification commands

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The data-dependent training experiment runs with `python -m src.reproduce --data data/2020_FPA_FOD_cons.csv` after the CSV is placed locally. Do not claim a training run succeeded unless you ran it or verified the attached results.

### Code and data policy

- Do not commit the raw FPA FOD CSV, user-specific paths, model binaries, credentials, or unlicensed lecture slides.
- Fit preprocessing and hyperparameters using training/validation data; do not tune on the test set.
- Maintain the approved feature audit. If a predictor may use information collected after discovery, add it only to clearly marked retrospective experiments.
- Cite dataset and method sources and record ChatGPT or other GenAI assistance in the individual declaration.
- Prefer clear reusable functions and tests over expanding one monolithic training file.

### PR checklist

- [ ] Clear purpose and linked issue
- [ ] Tests run, or reason documented if not run
- [ ] Results or screenshots where applicable
- [ ] Documentation updated
- [ ] No raw data or credentials committed
- [ ] Teammate reviewed before merge
