import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models.gpt_oss import apply_gpt_oss_expert_filter


def test_gpt_oss_expert_filter_uses_strict_threshold_and_renormalizes():
    topk_weights = torch.tensor(
        [
            [0.55, 0.30, 0.10, 0.05],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [
            [7, 12, 3, 25],
            [1, 2, 3, 4],
        ],
        dtype=torch.int32,
    )
    router_logits = torch.empty((2, 128), dtype=torch.float32)

    out = apply_gpt_oss_expert_filter(
        StandardTopKOutput(topk_weights, topk_ids, router_logits),
        threshold=0.10,
    )

    assert torch.equal(
        out.topk_ids,
        torch.tensor(
            [
                [7, 12, 3, -1],
                [1, 2, 3, 4],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        out.topk_weights[0],
        torch.tensor([0.55 / 0.95, 0.30 / 0.95, 0.10 / 0.95, 0.0]),
    )
    assert torch.allclose(out.topk_weights[1], topk_weights[1])
    assert torch.allclose(out.topk_weights.sum(dim=-1), torch.ones(2))


def test_gpt_oss_expert_filter_disabled_returns_original_output():
    topk_output = StandardTopKOutput(
        torch.tensor([[0.7, 0.3]], dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.empty((1, 128), dtype=torch.float32),
    )

    assert apply_gpt_oss_expert_filter(topk_output, threshold=0.0) is topk_output
