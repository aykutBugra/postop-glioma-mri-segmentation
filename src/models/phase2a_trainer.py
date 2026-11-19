from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


PHASE2A_CLASS_SAMPLING_WEIGHTS: Dict[int, float] = {
    1: 0.45,
    4: 0.35,
    3: 0.10,
    2: 0.10,
}

PHASE2A_WEIGHTED_CE_CLASS_WEIGHTS: Tuple[float, float, float, float, float] = (
    1.0,
    2.5,
    1.0,
    1.25,
    2.0,
)


class Phase2AClassAwareDataLoader(nnUNetDataLoader):
    """Foreground oversampling that prefers label_1 and label_4 without changing the rest of nnU-Net."""

    def __init__(
        self,
        *args,
        class_sampling_weights: Optional[Dict[int, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_sampling_weights = dict(class_sampling_weights or {})

    @staticmethod
    def _coerce_label(value: object) -> Optional[int]:
        if isinstance(value, (int, np.integer)):
            return int(value)
        return None

    def _select_weighted_foreground_class(
        self, class_locations: Optional[dict]
    ) -> Optional[Union[int, Tuple[int, ...]]]:
        if not class_locations:
            return None

        candidates: List[Union[int, Tuple[int, ...]]] = []
        weights: List[float] = []
        for raw_key, voxels in class_locations.items():
            if len(voxels) == 0:
                continue
            label = self._coerce_label(raw_key)
            if label is None:
                continue
            weight = self.class_sampling_weights.get(label)
            if weight is None or weight <= 0:
                continue
            candidates.append(raw_key)
            weights.append(float(weight))

        if not candidates:
            return None

        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        selected_idx = int(np.random.choice(len(candidates), p=probabilities))
        return candidates[selected_idx]

    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: Optional[dict],
        overwrite_class: Optional[Union[int, Tuple[int, ...]]] = None,
        verbose: bool = False,
    ):
        if force_fg and overwrite_class is None:
            overwrite_class = self._select_weighted_foreground_class(class_locations)
        return super().get_bbox(
            data_shape,
            force_fg,
            class_locations,
            overwrite_class=overwrite_class,
            verbose=verbose,
        )


class nnUNetTrainerPhase2ABase(nnUNetTrainer):
    """Shared base for Phase 2A Experiment A/B.

    Only two behaviors differ between experiments:
    1. class-aware foreground patch sampling
    2. CE class weights on/off
    """

    EXPERIMENT_MODE = "A"
    USE_WEIGHTED_CE = False
    SAMPLER_PROFILE = "phase2a_label1_label4_focus_v1"
    LOSS_PROFILE = "baseline_dice_ce"
    CLASS_SAMPLING_WEIGHTS = PHASE2A_CLASS_SAMPLING_WEIGHTS
    CE_CLASS_WEIGHTS = PHASE2A_WEIGHTED_CE_CLASS_WEIGHTS

    def initialize(self):
        super().initialize()
        self.phase2a_runtime_config = {
            "experiment_mode": self.EXPERIMENT_MODE,
            "sampler_profile": self.SAMPLER_PROFILE,
            "loss_profile": self.LOSS_PROFILE,
            "class_sampling_weights": self.CLASS_SAMPLING_WEIGHTS,
            "weighted_ce_enabled": self.USE_WEIGHTED_CE,
            "ce_class_weights": list(self.CE_CLASS_WEIGHTS)
            if self.USE_WEIGHTED_CE
            else None,
        }
        self.print_to_log_file(
            "Phase 2A runtime config:",
            self.phase2a_runtime_config,
            also_print_to_console=True,
        )

    def _get_phase2a_ce_weights(self) -> torch.Tensor:
        return torch.tensor(
            self.CE_CLASS_WEIGHTS, dtype=torch.float32, device=self.device
        )

    def _build_loss(self):
        if self.label_manager.has_regions or not self.USE_WEIGHTED_CE:
            return super()._build_loss()

        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {"weight": self._get_phase2a_ce_weights()},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = Phase2AClassAwareDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            class_sampling_weights=self.CLASS_SAMPLING_WEIGHTS,
        )
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerPhase2AExperimentA(nnUNetTrainerPhase2ABase):
    EXPERIMENT_MODE = "A"
    USE_WEIGHTED_CE = False
    LOSS_PROFILE = "baseline_dice_ce"


class nnUNetTrainerPhase2AExperimentB(nnUNetTrainerPhase2ABase):
    EXPERIMENT_MODE = "B"
    USE_WEIGHTED_CE = True
    LOSS_PROFILE = "baseline_dice_plus_weighted_ce"
