import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Paired-prompt training runs the same observation/action flow coordinate under
    # the canonical prompt and a semantically equivalent prompt in one 2B forward.
    paired_prompt_training: bool = False
    consistency_loss_weight: float = 0.0
    # symmetric_fm supervises both prompt branches with the shared FM target.
    # teacher_consistency anchors only the canonical branch with FM and trains the
    # paraphrase branch to match a stop-gradient canonical velocity prediction.
    paired_loss_mode: str = "symmetric_fm"

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.consistency_loss_weight < 0:
            raise ValueError("consistency_loss_weight must be non-negative")
        if self.consistency_loss_weight > 0 and not self.paired_prompt_training:
            raise ValueError("A positive consistency_loss_weight requires paired_prompt_training=True")
        if self.paired_loss_mode not in {"symmetric_fm", "teacher_consistency"}:
            raise ValueError(f"Unsupported paired_loss_mode: {self.paired_loss_mode}")
        if self.paired_loss_mode != "symmetric_fm" and not self.paired_prompt_training:
            raise ValueError("A non-default paired_loss_mode requires paired_prompt_training=True")
        if self.paired_loss_mode == "teacher_consistency" and self.consistency_loss_weight == 0:
            raise ValueError("teacher_consistency requires a positive consistency_loss_weight")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_prompt_alt=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.paired_prompt_training
                    else None
                ),
                tokenized_prompt_alt_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool)
                    if self.paired_prompt_training
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_action_expert_only_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the complete vision-language prefix while training the full action expert.

        Pi0 stores the PaliGemma and action-expert transformer parameters in a
        single ``PaliGemma/llm`` tree. Expert-specific leaves carry an ``_1``
        component suffix (for example ``mlp_1`` and ``q_einsum_1``). The
        projections outside ``PaliGemma`` are intentionally left trainable.
        """
        paligemma_params_filter = nnx_utils.PathRegex("PaliGemma/.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        return nnx.All(paligemma_params_filter, nnx.Not(action_expert_params_filter))

    def get_vlm_lora_full_action_expert_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train language LoRA adapters together with the full action expert.

        This deliberately freezes SigLIP and the Gemma-2B base parameters. The
        only trainable leaves inside ``PaliGemma`` are Gemma-2B LoRA adapters
        and the full ``_1`` action-expert tree; action/state/time projections
        outside ``PaliGemma`` remain trainable.
        """
        if "lora" not in self.paligemma_variant:
            raise ValueError("VLM LoRA + full Action Expert requires a LoRA PaliGemma variant")
        if "lora" in self.action_expert_variant:
            raise ValueError("VLM LoRA + full Action Expert requires a non-LoRA action expert variant")

        paligemma_params_filter = nnx_utils.PathRegex("PaliGemma/.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        lora_params_filter = nnx_utils.PathRegex(".*lora.*")
        return nnx.All(
            paligemma_params_filter,
            nnx.Not(action_expert_params_filter),
            nnx.Not(lora_params_filter),
        )
