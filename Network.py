"""
Chess-adapted version of the AlphaZero-style network (originally built for Go).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + x
        out = F.relu(out)
        return out


class AZNetChess(nn.Module):
    # 64 origin squares x 73 move types (56 queen-like + 8 knight + 9 underpromotion)
    NUM_MOVE_TYPES = 73

    def __init__(self, in_planes, channels, blocks, action_size, board_size=8, num_groups=8):
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
        self.blocks = nn.Sequential(*[ResidualBlock(channels, num_groups) for _ in range(blocks)])

        # policy head: conv straight to move-type channels, no Linear layer
        self.policy_conv = nn.Conv2d(channels, self.NUM_MOVE_TYPES, kernel_size=1, bias=False)
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=self.NUM_MOVE_TYPES)

        # value head (unchanged from the Go version)
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
        # which must match how your move-to-index lookup table is built.
        p = self.policy_conv(out)
        p = self.policy_gn(p)
        p = F.relu(p)
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
    IN_PLANES = 18  # placeholder -- set this to match your final board encoder

    # Leaner than an 8/128 config: starting point for unbatched, single-leaf
    # self-play on a single consumer GPU. Revisit once leaf evaluation is batched.
    net = AZNetChess(in_planes=IN_PLANES, channels=96, blocks=6, action_size=ACTION_SIZE)
    net.apply(init_weights)

    dummy = torch.randn(4, IN_PLANES, 8, 8)
    policy, value = net(dummy)
    assert policy.shape == (4, ACTION_SIZE)
    assert value.shape == (4,)
    print(sum(p.numel() for p in net.parameters()), "parameters")

    