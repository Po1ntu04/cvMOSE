"""Adapter for using first-frame GT masks with the SAM 3.1 video predictor.

SAM 3.1's public ``handle_request(add_prompt)`` API accepts text/point/box
prompts.  The MOSE homework, however, gives exact first-frame instance masks,
which is the natural VOS prompt used by the SAM2 homework script.  This module
keeps that impedance-matching in one small adapter: the caller still starts a
normal SAM 3.1 session, then this adapter injects the first-frame GT masks into
SAM 3.1's internal multiplex/SAM2-style tracker and exposes a simple per-frame
iterator.

The adapter deliberately avoids patching the facebookresearch/sam3 package.  It
uses private SAM 3.1 internals, so all such calls live here instead of being
scattered through the MOSE inference script.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence
import time
import uuid

import numpy as np
import torch


@dataclass(frozen=True)
class Sam31FrameOutput:
    """Normalized per-frame output from SAM 3.1 masklet propagation."""

    frame_index: int
    out_obj_ids: np.ndarray
    out_binary_masks: np.ndarray
    out_probs: np.ndarray | None = None
    raw_scores: np.ndarray | None = None
    source: str = "sam31_tracker"


class Sam31GtMaskAdapter:
    """Inject exact first-frame GT masks into a SAM 3.1 multiplex predictor.

    Public surface used by the assignment script:
      1. ``start_session(video_dir)``
      2. ``add_first_frame_masks(session_id, masks_by_obj_id)``
      3. ``propagate_forward(session_id)``
      4. ``close_session(session_id)``

    The internals are intentionally centralized here because the mask path is
    private in SAM 3.1: the official user API does not expose a first-frame mask
    prompt, but the bundled tracker still has mask-conditioning methods.
    """

    def __init__(self, predictor, *, run_mem_encoder: bool = True):
        self.predictor = predictor
        self.model = predictor.model
        self.run_mem_encoder = run_mem_encoder
        world_size = int(getattr(self.model, "world_size", 1))
        if world_size != 1:
            raise NotImplementedError(
                "Sam31GtMaskAdapter currently targets one visible CUDA device. "
                f"Got SAM3 world_size={world_size}."
            )

    @classmethod
    def build(
        cls,
        *,
        sam3_root: Path,
        checkpoint_path: Path,
        use_fa3: bool = False,
        compile_model: bool = False,
        max_num_objects: int = 64,
        multiplex_count: int = 16,
        async_loading_frames: bool = False,
        run_mem_encoder: bool = True,
    ) -> "Sam31GtMaskAdapter":
        """Build SAM 3.1 predictor from a local repo/checkpoint and wrap it."""
        import sys

        sys.path.insert(0, str(sam3_root))
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            compile=compile_model,
            warm_up=False,
            async_loading_frames=async_loading_frames,
        )
        predictor.model.eval()
        return cls(predictor, run_mem_encoder=run_mem_encoder)

    def start_session(
        self,
        video_dir: Path | str,
        *,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
    ) -> str:
        """Start a SAM 3.1 session and return its id.

        The shared SAM3 base predictor currently forwards ``offload_state_to_cpu``
        to all model ``init_state`` implementations, but the SAM 3.1 multiplex
        interactivity model does not accept that keyword.  To keep this adapter
        independent of upstream code edits, session registration is reproduced
        here with the kwargs accepted by SAM 3.1.
        """
        if offload_state_to_cpu:
            raise NotImplementedError(
                "SAM3.1 multiplex init_state does not expose offload_state_to_cpu; "
                "use the default false value for this adapter."
            )
        init_kwargs = {
            "resource_path": str(video_dir),
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self.predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.predictor.async_loading_frames
        inference_state = self.model.init_state(**init_kwargs)
        session_id = str(uuid.uuid4())
        self.predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return session_id

    def close_session(self, session_id: str) -> None:
        self.predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )

    def get_state(self, session_id: str) -> dict:
        return self.predictor._all_inference_states[session_id]["state"]

    @staticmethod
    def masks_from_label_map(
        label: np.ndarray, obj_ids: Sequence[int] | None = None
    ) -> dict[int, np.ndarray]:
        """Convert a single-channel label PNG array into {obj_id: bool mask}."""
        if label.ndim != 2:
            raise ValueError(f"Expected 2D label map, got shape={label.shape}")
        if obj_ids is None:
            obj_ids = [int(x) for x in np.unique(label) if int(x) != 0]
        masks: dict[int, np.ndarray] = {}
        for obj_id in obj_ids:
            mask = label == int(obj_id)
            if mask.any():
                masks[int(obj_id)] = mask
        if not masks:
            raise ValueError("No non-empty foreground masks found in label map")
        return masks

    def add_first_frame_masks(
        self,
        session_id: str,
        masks_by_obj_id: Mapping[int, np.ndarray],
        *,
        frame_idx: int = 0,
    ) -> Sam31FrameOutput:
        """Inject exact GT masks as conditioning masks at ``frame_idx``.

        The operation mirrors SAM2's ``add_new_mask`` semantics but uses SAM3.1's
        multiplex tracker state and metadata.
        """
        state = self.get_state(session_id)
        obj_ids = [int(x) for x in masks_by_obj_id.keys()]
        if len(obj_ids) != len(set(obj_ids)):
            raise ValueError(f"Duplicate object ids: {obj_ids}")
        if len(obj_ids) == 0:
            raise ValueError("masks_by_obj_id is empty")
        if len(obj_ids) > int(getattr(self.model, "max_num_objects", len(obj_ids))):
            raise ValueError(
                f"{len(obj_ids)} objects exceeds SAM3 max_num_objects="
                f"{getattr(self.model, 'max_num_objects', None)}"
            )

        # Validate shapes and stage masks on the SAM3 device.  ``_tracker_add_new_objects``
        # accepts floating masks and resizes them to the tracker's mask input size.
        first_shape = None
        mask_list: list[np.ndarray] = []
        for obj_id in obj_ids:
            mask = np.asarray(masks_by_obj_id[obj_id], dtype=bool)
            if mask.ndim != 2:
                raise ValueError(f"obj_id={obj_id}: expected 2D mask, got {mask.shape}")
            if not mask.any():
                raise ValueError(f"obj_id={obj_id}: empty mask")
            first_shape = mask.shape if first_shape is None else first_shape
            if mask.shape != first_shape:
                raise ValueError(
                    f"All masks must have same shape; got {mask.shape} vs {first_shape}"
                )
            mask_list.append(mask)

        device = getattr(self.model, "device", torch.device("cuda"))
        masks_t = torch.as_tensor(
            np.stack(mask_list, axis=0), dtype=torch.float32, device=device
        )
        obj_ids_np = np.asarray(obj_ids, dtype=np.int64)

        with torch.inference_mode():
            # Populate SAM3.1 image/propagation features for this conditioning frame.
            self.model._prepare_backbone_feats(state, frame_idx, reverse=False)

            # Add all first-frame GT objects in one multiplex batch.  This is the
            # closest analogue to SAM2's add_new_mask loop and preserves full mask
            # information rather than reducing it to points/boxes.
            state["sam2_inference_states"] = self.model._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=state["num_frames"],
                new_obj_ids=obj_ids_np,
                new_obj_masks=masks_t,
                tracker_states_local=state["sam2_inference_states"],
                orig_vid_height=state["orig_height"],
                orig_vid_width=state["orig_width"],
                feature_cache=state["feature_cache"],
            )

            self._install_gt_metadata(state, obj_ids, frame_idx)

            # Cache exact GT masks for the prompted frame.  We still save the
            # original palette PNG in the assignment script, but this gives the
            # adapter a coherent state snapshot and trace output.
            if getattr(self.model, "rank", 0) == 0:
                gt_obj_to_mask = {
                    obj_id: torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    for obj_id, mask in zip(obj_ids, mask_list)
                }
                self.model._cache_frame_outputs(state, frame_idx, gt_obj_to_mask)

            self.model.add_action_history(
                state, action_type="add", frame_idx=frame_idx, obj_ids=obj_ids
            )

        return Sam31FrameOutput(
            frame_index=frame_idx,
            out_obj_ids=obj_ids_np.copy(),
            out_binary_masks=np.stack(mask_list, axis=0).astype(bool),
            out_probs=np.ones(len(obj_ids), dtype=np.float32),
            raw_scores=np.ones(len(obj_ids), dtype=np.float32),
            source="gt_mask_prompt",
        )

    def _install_gt_metadata(self, state: dict, obj_ids: Sequence[int], frame_idx: int) -> None:
        """Install top-level metadata expected by SAM3.1 partial propagation."""
        metadata = self.model._initialize_metadata()
        obj_ids_np = np.asarray([int(x) for x in obj_ids], dtype=np.int64)
        metadata["obj_ids_per_gpu"][0] = obj_ids_np.copy()
        metadata["obj_ids_all_gpu"] = obj_ids_np.copy()
        metadata["num_obj_per_gpu"][0] = len(obj_ids_np)
        metadata["max_obj_id"] = int(obj_ids_np.max()) if len(obj_ids_np) else -1
        metadata["obj_id_to_score"] = {int(obj_id): 1.0 for obj_id in obj_ids_np}
        for obj_id in obj_ids_np:
            metadata["obj_id_to_sam2_score_frame_wise"][frame_idx][int(obj_id)] = (
                torch.tensor(1.0, dtype=torch.float32, device=self.model.device)
            )

        rank0_metadata = metadata.get("rank0_metadata")
        if rank0_metadata and "masklet_confirmation" in rank0_metadata:
            # Mark GT-conditioned masklets as confirmed so post-processing will not
            # hide them as detector-unconfirmed masklets.
            rank0_metadata["masklet_confirmation"]["status"] = np.full(
                len(obj_ids_np), 2, dtype=np.int64
            )
            rank0_metadata["masklet_confirmation"]["consecutive_det_num"] = np.full(
                len(obj_ids_np),
                int(getattr(self.model, "masklet_confirmation_consecutive_det_thresh", 1)),
                dtype=np.int64,
            )

        if getattr(self.model, "is_multiplex", False):
            count_buckets = getattr(self.model, "_count_buckets_in_states", None)
            if count_buckets is not None:
                num_buc = int(count_buckets(state["sam2_inference_states"]))
            else:
                num_buc = len(state["sam2_inference_states"])
            metadata["num_buc_per_gpu"][0] = num_buc
            metadata["gpu_metadata"] = {
                "N_obj": len(obj_ids_np),
                "obj_first_frame": torch.full(
                    (len(obj_ids_np),), frame_idx, dtype=torch.long, device=self.model.device
                ),
                "consecutive_unmatch_count": torch.zeros(
                    len(obj_ids_np), dtype=torch.long, device=self.model.device
                ),
                "trk_keep_alive": torch.ones(
                    len(obj_ids_np), dtype=torch.bool, device=self.model.device
                ),
                "removed_mask": torch.zeros(
                    len(obj_ids_np), dtype=torch.bool, device=self.model.device
                ),
                "overlap_pair_counts": torch.zeros(
                    (len(obj_ids_np), len(obj_ids_np)), dtype=torch.long, device=self.model.device
                ),
                "last_occluded_tensor": torch.zeros(
                    len(obj_ids_np), dtype=torch.long, device=self.model.device
                ),
            }

        state["tracker_metadata"] = metadata

    def propagate_forward(
        self,
        session_id: str,
        *,
        start_frame_idx: int = 0,
        max_frame_num_to_track: int | None = None,
    ) -> Iterator[Sam31FrameOutput]:
        """Propagate GT-conditioned masklets forward and yield normalized outputs."""
        state = self.get_state(session_id)
        tracker_states = state["sam2_inference_states"]
        if not tracker_states:
            raise RuntimeError("No SAM3.1 tracker states; call add_first_frame_masks first")

        num_frames = int(state["num_frames"])
        end_exclusive = num_frames
        if max_frame_num_to_track is not None:
            end_exclusive = min(num_frames, start_frame_idx + max_frame_num_to_track)

        with torch.inference_mode():
            # Ensure conditioning-frame outputs are consolidated into memory before
            # stepping frame-by-frame.
            for tracker_state in tracker_states:
                self.model.tracker.propagate_in_video_preflight(
                    tracker_state, run_mem_encoder=self.run_mem_encoder
                )

            for frame_idx in range(start_frame_idx, end_exclusive):
                if frame_idx == start_frame_idx and frame_idx in state.get(
                    "cached_frame_outputs", {}
                ):
                    # Return the exact GT-conditioned frame from cache.
                    obj_id_to_mask = state["cached_frame_outputs"][frame_idx]
                    metadata = state["tracker_metadata"]
                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": metadata["obj_id_to_score"],
                        "obj_id_to_sam2_score": metadata[
                            "obj_id_to_sam2_score_frame_wise"
                        ][frame_idx],
                    }
                    outputs = self.model._postprocess_output(state, out)
                    yield self._normalize_outputs(frame_idx, outputs, source="gt_mask_prompt")
                    continue

                self.model._prepare_backbone_feats(state, frame_idx, reverse=False)
                obj_ids, low_res_masks, raw_scores = (
                    self.model._propogate_tracker_one_frame_local_gpu(
                        tracker_states,
                        frame_idx=frame_idx,
                        reverse=False,
                        run_mem_encoder=self.run_mem_encoder,
                    )
                )
                raw_scores_np = raw_scores.detach().float().cpu().numpy()
                obj_id_to_mask = {}
                for row, obj_id in enumerate(obj_ids):
                    obj_id_to_mask[int(obj_id)] = self.model._convert_low_res_mask_to_video_res(
                        low_res_masks[row], state
                    )
                    state["tracker_metadata"]["obj_id_to_sam2_score_frame_wise"][
                        frame_idx
                    ][int(obj_id)] = raw_scores[row]

                out = {
                    "obj_id_to_mask": obj_id_to_mask,
                    "obj_id_to_score": state["tracker_metadata"]["obj_id_to_score"],
                    "obj_id_to_sam2_score": state["tracker_metadata"][
                        "obj_id_to_sam2_score_frame_wise"
                    ][frame_idx],
                }
                self.model._cache_frame_outputs(state, frame_idx, obj_id_to_mask)
                outputs = self.model._postprocess_output(state, out)
                yield self._normalize_outputs(
                    frame_idx, outputs, raw_scores=raw_scores_np, source="sam31_tracker"
                )

    @staticmethod
    def _normalize_outputs(
        frame_idx: int,
        outputs: Mapping[str, object],
        *,
        raw_scores: np.ndarray | None = None,
        source: str,
    ) -> Sam31FrameOutput:
        out_obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
        out_masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
        out_probs = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
        if out_masks.ndim == 2:
            out_masks = out_masks[None, ...]
        return Sam31FrameOutput(
            frame_index=int(frame_idx),
            out_obj_ids=out_obj_ids,
            out_binary_masks=out_masks,
            out_probs=out_probs,
            raw_scores=raw_scores,
            source=source,
        )
