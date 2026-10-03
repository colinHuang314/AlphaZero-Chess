# AlphaZero-Style Chess Bot

A chess AI following the AlphaZero approach: a PyTorch policy/value network guiding Monte Carlo Tree Search, pretrained on human games and then improved through self-play. It's an adaptation of my [AlphaZero-style Go bot](https://github.com/colinHuang314/AlphaZero-Style-Go-Bot) to chess.

## Overview
- **Network:** SE-ResNet (8 blocks, 64 channels) with GroupNorm over an 18-plane board encoding
- **Search:** batched MCTS with virtual loss, evaluating many leaves per GPU pass, with Syzygy tablebase lookups in the endgame
- **Training:** supervised pretraining on 27M+ positions (elite Lichess games plus tablebase endgames), then self-play, on an RTX 4060
- **Result:** about 1000 Elo. It plays human-like positional chess and once found a queen sacrifice leading to forced mate, but it struggles with sharp tactics, and network evaluation keeps search slow
- **Analysis UI:** a Lichess-style board in the browser with a live eval bar, candidate-move arrows and the search's top moves
- **See also:** [Minimax-Chess](https://github.com/colinHuang314/Minimax-Chess), the C# alpha-beta engine and 3D Unity game I built next

## What I debugged
Replay buffer mis-sizing, corrupted data from max-length games, temperature-schedule bugs, and a value head that appeared to overpower the policy.

## Development notes
The chess engine adapts my original Go engine, which I wrote myself from the AlphaGo Zero paper. The chess adaptation was built with AI-assisted development (Claude); I designed the architecture and training pipeline and debugged and evaluated the models.

## Try it
Weights: [Releases](https://github.com/colinHuang314/AlphaZero-Chess/releases/tag/v1.0)

```bash
pip install -r requirements.txt

# put model_human_pretrained_7-7.pt in Models/, then (opens http://localhost:5000)
python AnalyzeUI2.py
```

Optional: Syzygy 3-4-5 piece tablebases in `syzygy/Syzygy345WDL` and `syzygy/Syzygy345DTZ` enable endgame lookups; without them the search uses the network's evaluation everywhere.

| File | What |
|---|---|
| `TrainingLoop3.py` | self-play training loop with arena gating |
| `TrainHuman.py`, `HumanData.py`, `Tablebasedata.py` | supervised pretraining on human games and tablebase endgames |
| `mcts_core.py` | batched MCTS with virtual loss and tablebase probing |
| `Network2.py`, `Encoder.py` | network and board/move encoding |
| `AnalyzeUI2.py` | browser analysis UI |
| `Arena.py`, `SpeedTest.py` | model-vs-model matches, search benchmarks |
