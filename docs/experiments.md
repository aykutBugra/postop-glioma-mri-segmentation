# Experiment development and decisions

| Experiment | Question | Result |
| --- | --- | --- |
| Longitudinal mask prototype | Can serial measurements be extracted from existing masks and joined to a follow-up timeline? | Produced tumour and cavity volumes and a Streamlit prototype. |
| Binary 3D U-Net | Can MR images yield an automatic foreground mask? | Established an automatic volume baseline. Later iterations selected the checkpoint with full-volume validation and swept thresholds separately. |
| Custom multiclass U-Net | Can automatic masks yield distinct tissue and cavity volumes? | Improved derived volume measurement on the historical split. NETC and RC errors persisted despite class-aware crop, weighted sampling and loss variants. |
| Historical forecasting | Can previous volumes and other available features predict the next volume-change category? | Removed future-derived features after a leakage review. The remaining results were exploratory. The old partition called test was used to select forecasting tasks. |
| Phase 1F | What does a standard 3D nnU-Net v2 baseline achieve on fold 0? | Completed 1,000 epochs, recovered from a training interruption, and saved full-volume validation outputs for 96 timepoints. |
| Experiment A | Does class-aware foreground patch selection improve NETC and RC? | Foreground Dice improved slightly. NETC and RC active-case Dice fell and foreground volume MAE exceeded the acceptance threshold. |
| Experiment B | Does weighted cross-entropy improve the class-aware sampler? | The log ends at the start of epoch index 210. Peak logged EMA pseudo Dice was 0.8697 at epoch index 184. No completed full-volume assessment is available. |

## Phase 2A design

Experiment A used the same fold, model and Dice plus cross-entropy loss as Phase 1F. For a forced foreground crop, it sampled from the present classes in proportion to weights of 0.45 NETC, 0.35 RC, 0.10 ET and 0.10 SNFH. Experiment B enabled cross-entropy class weights of 1.0, 2.5, 1.0, 1.25 and 2.0 in background, NETC, SNFH, ET and RC order. [Trainer implementation](../src/models/phase2a_trainer.py)

## Evaluator conventions

The original Phase 2A analysis writes a field named `mean_abs_wt_volume_error_ml` using all foreground labels, including RC. The release labels this as foreground volume MAE and reports a separate WT MAE for labels 1–3.

The original `lt_1ml` subgroup includes zero-volume reference cases. The release subgroup checks used `0 < reference volume < 1 mL`. On this definition, Phase 1F versus A gave NETC Dice of 0.4544 versus 0.4415 over 23 cases, and RC Dice of 0.3479 versus 0.3607 over 24. Absence is a separate false-positive question. [Historical evaluator code](../src/eval/phase2a_validation_analysis.py) · [Aggregate results](../results/README.md)

## Split and provenance

The Phase 1F and A manifests contain the same fold-0 validation timepoints and no patient overlap among the three partitions. The planner nevertheless estimated its fingerprint using all 594 cases. The 30 reserved patients were also in the earlier custom U-Net test partition. The reported Phase 1F and A figures therefore describe the shared historical validation fold.
