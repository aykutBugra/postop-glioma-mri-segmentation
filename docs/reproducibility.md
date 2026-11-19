# Reproducing the reference workflow

## Aggregate figures

The public CSVs are sufficient to render every README plot. From the project root, install `requirements.txt` with Python 3.12 and run `python scripts/plot_results.py`. Each run regenerates the four PNGs in `results/figures/`. [Aggregate inputs and metric definitions](../results/README.md)

## Local data preparation

Download the TCIA NIfTI collection and the clinical and segmentation-volume spreadsheets. Keep the inputs local. The names below are the defaults consumed by the original preparation scripts. The MRI files must retain the suffixes from the collection, including `_brain_t2f.nii.gz` for T2-FLAIR and `_tumorMask.nii.gz` for the segmentation. [MU-Glioma-Post download and data license](https://www.cancerimagingarchive.net/collection/mu-glioma-post/)

```text
data/
├── raw/
│   └── PatientID_<id>/
│       └── Timepoint_<visit>/
│           ├── <case>_brain_t1c.nii.gz
│           ├── <case>_brain_t1n.nii.gz
│           ├── <case>_brain_t2w.nii.gz
│           ├── <case>_brain_t2f.nii.gz
│           └── <case>_tumorMask.nii.gz
├── metadata/
│   ├── MU-Glioma-Post_ClinicalData-July2025.xlsx
│   └── MU-Glioma-Post_Segmentation_Volumes.xlsx
└── sample/
```

Create the private directories from the project root. Run the preparation commands in order with the dependencies installed.

```bash
mkdir -p data/raw data/metadata data/sample outputs
python -m src.data.build_dataset_index --project-root .
python -m src.data.normalize_metadata_links --project-root .
python -m src.models.prepare_nnunet_dataset --project-root . --dataset-id 901
python -m src.models.build_splits --project-root . --seed 42
```

The optional `python -m src.data.inspect_dataset_structure --project-root .` command summarizes the file tree before conversion. The preserved `scripts/env_check.py` is a historical inventory check. It expects a local README and a PDF in `data/metadata/` in addition to the spreadsheets. Those two files are optional for the preparation commands above. Preparation writes indexes to `outputs/phase1_dataset_contract/` and the converted dataset and split files to `outputs/phase1_nnunet_dataset/`. Conversion may copy files if hardlinking is unavailable. Give it adequate disk space and use a separate workspace for each dataset. The command replaces converted files in its own generated dataset folder when rerun.

## Training and validation

Use a Linux environment with a CUDA GPU, `rsync`, directory symlink support, and an isolated Jupyter kernel. Install Jupyter and `ipykernel` and set up PyTorch with the CUDA build appropriate for the GPU. Install an nnU-Net v2 release compatible with that environment, and confirm `nnUNetv2_plan_and_preprocess`, `nnUNetv2_train` and `nnUNetv2_predict` are on the kernel’s PATH. The original logs identify an NVIDIA A100 40 GB, PyTorch `2.10.0+cu128` and the standard `nnUNetPlans` planner. The exact nnU-Net package version was not recorded. Re-evaluate compatibility and resolve any dependency conflicts in an isolated GPU environment before attempting a run. [nnU-Net paper and method](https://doi.org/10.1038/s41592-020-01008-z)

1. Open [Phase 1F reference](../notebooks/phase1F_nnunet_train.ipynb) with the prepared local dataset. Start the kernel in the project root or `notebooks/`, or set `GLIOMA_PROJECT_ROOT` to the project directory. The notebook sets nnU-Net’s three working paths under `outputs/`.
2. Use a new `RUN_NAME` for a fresh run. Use resume mode for an existing run. The notebook stages the converted dataset, calls `nnUNetv2_plan_and_preprocess -d 901 -c 3d_fullres`, installs the one-fold split file, and runs `nnUNetv2_train 901 3d_fullres 0`. The historical workflow fingerprints all converted cases, including the reserved partition.
3. Run the validation section with the local nnU-Net results and the validation case files. It runs inference on the validation volumes and uses the best checkpoint when available. This requires the local model folder with its plans, dataset JSON and checkpoint. Eighty-four reserved cases are absent from the published metrics. Case-level tables, predictions and overlays remain under the ignored `outputs/` directory.
4. Open [Phase 2A ablation](../notebooks/phase2A_nnunet_train.ipynb) for the optional custom sampler. It registers the preserved trainer inside the local nnU-Net environment. Use an isolated environment that permits that write, and make the Phase 1F evaluation available locally for comparison. The notebook’s original Phase 2A evaluator retains the historical subgroup and volume naming conventions. See the notes before interpreting those outputs.

The notebooks are reference workflows. Their GPU cells, full-data preparation, and checkpoint inference were not run for this package. A fresh independent evaluation would require a planner fit on training data only, validation for model selection, and a cohort that did not inform prior development. [Experiment notes](experiments.md) · [Recorded settings](../configs/phase1f.reference.yaml)

## Package verification

The data preparation commands were run on a synthetic fixture with 10 patients and 20 timepoints. They produced 80 image channels, 20 masks and disjoint patient partitions. The evaluation contract CLI and aggregate plot generator also ran successfully. Both notebooks passed Python syntax checks with saved outputs removed. The synthetic fixture is kept outside the release.
