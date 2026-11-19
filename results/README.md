# Aggregate results

These files summarize archived Phase 1F and Experiment A runs. They contain cohort-level or epoch-level values.

| File | Contents and source |
| --- | --- |
| [validation_dice.csv](validation_dice.csv) | Foreground and four label Dice summaries for each run. Recomputed from each run’s `validation_eval/validation_metrics_per_case.csv` and checked against the corresponding saved label and validation summaries. |
| [foreground_dice_histogram.csv](foreground_dice_histogram.csv) | Fixed-width Phase 1F bins of 0.05, left-inclusive and right-exclusive except for the final bin, which includes 1.0. Counts sum to 96. |
| [training_curves.csv](training_curves.csv) | Logged train loss, online validation loss, and the arithmetic mean of four logged pseudo Dice values at each of 1,000 epochs per run. Phase 1F combines the first training log and the resumed run. Epoch indices are zero-based. The figure uses one-based labels and overlays trailing 25-epoch means. |

The archived run names are `phase1f_fold0_full_run_001` and `phase2a_exp_a_run_001`. Each full-volume table comes from `checkpoint_best.pth` on the same 96 timepoints. Foreground Dice uses the union of nonzero labels. Label Dice averages across cases where the label exists in the reference, prediction, or both. Both-empty label masks are excluded. Quartiles use linear interpolation between observed values. [Plot generator](../scripts/plot_results.py)
