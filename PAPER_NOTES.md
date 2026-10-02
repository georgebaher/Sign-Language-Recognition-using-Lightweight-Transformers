# Paper notes

Use [the results table](runs/seed379/results.md) for scores and parameter counts.
Audits and per-run records under `runs/seed379/` hold the supporting data.

- AVASAG is strongly class-imbalanced. Accuracy and macro-F1 can tell different
  stories; report both.
- One seed does not establish training stability. Small score differences are
  descriptive until uncertainty is assessed.
- Learned fusion weights describe score combination, not linguistic importance.
- FG2027 aligns modalities by frame and selects one person per frame. The earlier
  loader used row order, so its scores are not directly interchangeable with these.
- Do not cite the unpublished IberSPEECH paper. Keep routine data-handling details
  and training curves out of the manuscript; retain the supplied model diagrams.
