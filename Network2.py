"""
Chess network, right-sized for single-consumer-GPU self-play.

Design notes (why this differs from the Go-derived original):
  - Chess is 8x8, not 19x19. Each residual block (two 3x3 convs) grows the
    receptive field radius by 2, so even 6 blocks already "sees" ~25 squares
    across -- more than the whole board. Extra depth beyond ~8 blocks buys
    almost nothing positionally on a board this small; it just costs GPU time.
  - Channel count (feature richness per square) matters more than depth here,
    so this trades some of the old channel budget for a squeeze-excitation
    (SE) block per residual layer instead -- cheap (a couple of tiny FC layers
    on globally-pooled features), and this is the single change Leela Chess
    Zero's own ablations found gave the best strength-per-compute of any
    architecture tweak they tried on this exact family of network.
  - The policy head previously ended with GroupNorm + ReLU directly on the
    logits, right before softmax. That's a bug, not a stylistic choice: ReLU
    clips every negative logit to exactly 0, so any two moves the network
    currently dislikes become indistinguishable to the softmax. Standard
    AlphaZero-style policy heads output raw logits with no final activation
    -- fixed here.
    
NOTE: this is a different architecture, not weight-compatible with old
checkpoints (different SE parameters, no post-policy-conv norm/activation).
Expect to retrain from human-pretraining, not to load old .pt files into it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class SqueezeExcite(nn.Module):
    """Global-context channel gate. Pools each channel to a single number,
    runs a tiny bottleneck FC, and rescales the channel-wise feature maps.
    Adds ~channels^2/reduction extra params/flops -- negligible next to the
    3x3 convs it sits beside, but gives the block a cheap way to reason about
    the whole board at once (useful for things like "is my king safe
    anywhere" that a purely local 3x3 conv can't see in one layer)."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))                # (B, C) global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))         # (B, C) gate in (0, 1)
        return x * s.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=8, se_reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.se = SqueezeExcite(channels, reduction=se_reduction)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)
        out = out + x
        out = F.relu(out)
        return out


class AZNetChess(nn.Module):
    # 64 origin squares x 73 move types (56 queen-like + 8 knight + 9 underpromotion)
    NUM_MOVE_TYPES = 73

    def __init__(self, in_planes, channels, blocks, action_size, board_size=8,
                 num_groups=8, se_reduction=4):
        super().__init__()
        assert action_size == board_size * board_size * self.NUM_MOVE_TYPES, (
            "action_size must equal board_size * board_size * NUM_MOVE_TYPES "
            "for the conv-based policy head"
        )
        self.board_size = board_size
        self.action_size = action_size

        # input block
        self.conv_in = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        # residual tower
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels, num_groups, se_reduction) for _ in range(blocks)
        ])

        # policy head: conv straight to move-type channels -> raw logits.
        # No norm/activation after this conv -- softmax over legal moves
        # needs unclipped logits to tell "bad" from "very bad" apart.
        self.policy_conv = nn.Conv2d(channels, self.NUM_MOVE_TYPES, kernel_size=1, bias=True)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.gn_in(out)
        out = F.relu(out)
        out = self.blocks(out)

        # policy head -> (B, 73, 8, 8) -> (B, 8, 8, 73) -> (B, 4672)
        # flat index = (rank * board_size + file) * NUM_MOVE_TYPES + move_type,
        # matching your existing move-to-index lookup table -- unchanged from
        # the original network, so no Encoder.py changes needed.
        p = self.policy_conv(out)
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), -1)

        # value head
        v = self.value_conv(out)
        v = self.value_gn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v


if __name__ == "__main__":
    ACTION_SIZE = 8 * 8 * AZNetChess.NUM_MOVE_TYPES  # 4672
    IN_PLANES = 18

    # 64 channels x 8 blocks + SE: narrower than the old 96x6, one block
    # deeper. Channel count dominates conv compute quadratically, so
    # (64/96)^2 * (8/6) ~= 0.59x the tower compute of the old net, despite
    # the extra block -- plus SE's own overhead, which is small (a couple
    # of tiny FC layers on pooled features per block).
    net = AZNetChess(in_planes=IN_PLANES, channels=64, blocks=8, action_size=ACTION_SIZE)
    net.apply(init_weights)

    dummy = torch.randn(4, IN_PLANES, 8, 8)
    policy, value = net(dummy)
    assert policy.shape == (4, ACTION_SIZE)
    assert value.shape == (4,)
    print(sum(p.numel() for p in net.parameters()), "parameters")