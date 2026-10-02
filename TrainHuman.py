"""
Supervised pretraining on human games. This is a sanity check for the
network + encoder -- not "pure AlphaZero" (that's self-play only) -- to
confirm the architecture can actually learn chess patterns before investing
time in the slower, currently-unbatched self-play loop.

Policy targets here are a single move index per position (the move actually
played), not a full ACTION_SIZE-length one-hot vector the way the Go script
built target_pis. That distinction matters at this action-space size: 1M
positions x 4672 floats x 4 bytes is ~18.7GB as dense one-hot vectors, vs
~8MB as int64 indices. F.cross_entropy takes integer class indices directly,
so the dense vector never needs to exist.

Data loading streams to disk via build_dataset_to_disk() + MemmapChessDataset
instead of holding every encoded position in RAM (build_dataset(), still in
human_data.py, does that, and is fine for a few hundred thousand positions --
beyond that the in-RAM array itself, plus the transient doubling np.stack()
causes while assembling it, adds up fast: 20,000 games was already ~7.6GB;
going further raises actual GBs/disk space accordingly, just no longer as a
hard RAM ceiling). Random-access shuffling still works normally through
DataLoader -- np.memmap supports it directly, the OS just pages in only
what's actually read instead of everything up front.

Also updated torch.cuda.amp.autocast/GradScaler (deprecated as of torch 2.4)
to the current torch.amp API, which is also device-aware -- it cleanly
disables itself on CPU instead of erroring, rather than assuming CUDA.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from HumanData import build_dataset_to_disk
from Network2 import AZNetChess, init_weights
from Encoder import IN_PLANES, ACTION_SIZE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Models/, syzygy/, Human Data/ ... live next to this script


class MemmapChessDataset(Dataset):
    """Wraps the files written by build_dataset_to_disk() for random-access
    reading through DataLoader, without ever loading the full dataset into
    RAM at once."""

    def __init__(self, n_positions, out_prefix):
        self.n = n_positions
        self.out_prefix = out_prefix
        # Don't open memmaps here — only store paths/shapes
        self._states = None
        self._moves = None
        self._values = None

    def _init_memmaps(self):
        """Open memmaps lazily — called inside each worker after spawning."""
        if self._states is None:
            self._states = np.memmap(f"{self.out_prefix}_states.bin", dtype=np.float32, mode='r',
                                     shape=(self.n, IN_PLANES, 8, 8))
            self._moves  = np.memmap(f"{self.out_prefix}_moves.bin",  dtype=np.int64,   mode='r',
                                     shape=(self.n,))
            self._values = np.memmap(f"{self.out_prefix}_values.bin", dtype=np.float32, mode='r',
                                     shape=(self.n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._init_memmaps()  # no-op after first call in each worker
        assert self._states is not None and self._moves is not None and self._values is not None
        return (
            torch.from_numpy(np.array(self._states[idx])),
            torch.tensor(int(self._moves[idx])),
            torch.tensor(float(self._values[idx])),
        )

# Data
PGN_PATH = os.path.join(BASE_DIR, "Human Data", "lichess_elite_2022-01.pgn")
# CACHE_PREFIX = os.path.join(BASE_DIR, "Human Data", "human_data_cache")          # where build_dataset_to_disk writes its files
CACHE_PREFIX = os.path.join(BASE_DIR, "Human Data", "tablebase_endgames_combined")
MODEL_PREFIX = os.path.join(BASE_DIR, "Models", "model_human_pretrained")

MAX_GAMES = 20_000 * 20                          # raise freely now -- this scales with disk space, not RAM
MIN_ELO = None                             # e.g. 2300, only usable if the PGN has WhiteElo/BlackElo headers

# Model
CHANNELS = 64 # 96
BLOCKS = 8 # 6

# Optimizer / LR
LR = 1e-3
VALUE_LOSS_WEIGHT = 1.0
WEIGHT_DECAY = 1e-4
MAX_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

EPOCHS = 5
BATCH_SIZE = 256


if __name__ == "__main__":
    #print date and time
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start = time.time()

    CACHE_FILES = [
        f"{CACHE_PREFIX}_states.bin",
        f"{CACHE_PREFIX}_moves.bin",
        f"{CACHE_PREFIX}_values.bin",
    ]

    if all(os.path.exists(f) for f in CACHE_FILES):
        # Read the position count from the existing moves file
        n_positions = np.memmap(f"{CACHE_PREFIX}_moves.bin", dtype=np.int64, mode='r').shape[0]
        print(f"Using cached dataset: {n_positions:,} positions")
    else:
        n_positions = build_dataset_to_disk(PGN_PATH, CACHE_PREFIX, max_games=MAX_GAMES, min_elo=MIN_ELO)
        print(f"Built dataset: {n_positions:,} positions")

    dataset = MemmapChessDataset(n_positions, CACHE_PREFIX)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=4,                        # start here, tune down if RAM/CPU is tight
        persistent_workers=True,              # keeps workers alive between epochs (avoids respawn cost)
        prefetch_factor=2,                    # each worker pre-loads 2 batches ahead (default, explicit is clearer)
    )

    model = AZNetChess(in_planes=IN_PLANES, channels=CHANNELS, blocks=BLOCKS, action_size=ACTION_SIZE).to(DEVICE)
    model.apply(init_weights)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        for states_b, move_idx_b, target_v_b in loader:
            states_b = states_b.to(DEVICE, non_blocking=True)
            move_idx_b = move_idx_b.to(DEVICE, non_blocking=True)
            target_v_b = target_v_b.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
                pred_pis, pred_vs = model(states_b)
                policy_loss = F.cross_entropy(pred_pis, move_idx_b)
                value_loss = F.mse_loss(pred_vs, target_v_b)
                loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(opt)
            scaler.update()

        print(f"Epoch {epoch+1} | loss {loss.item():.4f} "
              f"| policy {policy_loss.item():.4f} "
              f"| value {value_loss.item():.4f} "
              f"| {time.time() - epoch_start:.1f}s")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, fr"{MODEL_PREFIX}_{int(time.time())}.pt")

    print(f"time taken: {time.time() - start:.1f}s")