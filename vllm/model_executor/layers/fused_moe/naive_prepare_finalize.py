# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    count_expert_num_tokens,
    moe_kernel_quantize_input,
)


class NaivePrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    A reference prepare/finalize implementation that emulates the legacy
    broadcast + all-reduce dispatch/combine path used for EP + DP without
    a dedicated all2all backend.
    """

    def __init__(self, num_local_experts: int):
        super().__init__()
        self.num_local_experts = num_local_experts

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        # Follow the default dtype from the routing op.
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        # The final tensor is still DP-sharded after combine.
        return False

    def _get_dp_local_sizes(self) -> list[int] | None:
        ctx = get_forward_context()
        if ctx.dp_metadata is None:
            return None
        return ctx.dp_metadata.get_chunk_sizes_across_dp_rank()

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        del num_experts  # Unused for the naive path.

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # Only top-1 routing is supported in this mode.
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        sizes = self._get_dp_local_sizes()

        # Gather tokens and routing decisions from every DP rank so that this
        # rank can execute its local experts for the entire global batch.
        dp_group = get_dp_group()
        if dp_group.world_size > 1:
            assert sizes is not None, (
                "DP metadata must provide chunk sizes for naive MoE dispatch"
            )
        if dp_group.world_size > 1:
            gathered = dp_group.all_gatherv(
                [a1, topk_weights, topk_ids],
                dim=0,
                sizes=sizes,
            )
            a1, topk_weights, topk_ids = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor], gathered
            )

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )

        expert_tokens_meta: mk.ExpertTokensMetadata | None = None
        if expert_map is not None:
            expert_num_tokens = count_expert_num_tokens(
                topk_ids, self.num_local_experts, expert_map
            )
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens,
                expert_num_tokens_cpu=expert_num_tokens.to("cpu", non_blocking=False),
            )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        reduced = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        sizes = self._get_dp_local_sizes()
        dp_group = get_dp_group()
        if dp_group.world_size > 1:
            assert sizes is not None, (
                "DP metadata must provide chunk sizes for naive MoE combine"
            )
            reduced = dp_group.reduce_scatterv(reduced, dim=0, sizes=sizes)

        output.copy_(reduced)
