# ChessBot

A chess engine in C# using bitboards, negamax + alpha-beta search, and a
tapered PeSTO-style evaluation (material + piece-square tables + mobility).

## Build & run

```bash
dotnet build -c Release
dotnet run -c Release            # UCI mode (plug into any UCI-compatible GUI, e.g. Arena, CuteChess, En Croissant)
dotnet run -c Release -- cli      # simple text CLI: type moves like e2e4, or "go" to have the bot move
```

To use with a GUI: point it at the built executable
(`bin/Release/net8.0/ChessBot(.exe)`) as a UCI engine.

## Architecture

- **Types.cs / Move.cs** — piece/color enums, a move packed into a single 32-bit int
  (from, to, flag, moved piece, captured piece).
- **Bitboards.cs** — precomputed knight/king/pawn attack tables, and ray-based
  sliding attack generation for bishops/rooks/queens (classical approach —
  correct and reasonably fast; can be upgraded to magic bitboards later for
  a further speed boost).
- **Magic.cs** — precomputed rook/bishop magic numbers and a helper,
  not yet wired into attack generation.
- **Board.cs** — bitboard board state, Zobrist hashing, make/unmake move with
  full undo info (castling rights, en passant, halfmove clock), null-move support.
- **MoveGen.cs** — full legal move generation (castling, en passant, promotions),
  legality checked via make/is-king-in-check/unmake.
- **Evaluation.cs** — tapered (midgame/endgame blended by game phase) material
  + piece-square tables (PeSTO-style) + a lightweight mobility term.
- **Search.cs** — iterative deepening negamax with alpha-beta pruning,
  quiescence search (captures only), a transposition table, MVV-LVA capture
  ordering, killer moves, history heuristic, and late move reductions (LMR).
- **TranspositionTable.cs** — fixed-size hash table keyed by Zobrist hash.
- **Uci.cs / Program.cs** — UCI protocol loop, plus a minimal CLI for quick
  manual testing without a GUI.

## Notes / things worth improving next

- **Magic bitboards** would meaningfully speed up sliding-piece attacks over
  the current ray-scanning approach. The magic numbers are already in
  `Magic.cs`; what's left is building the lookup tables and switching
  `Attacks` over to them.
- **Repetition detection** (threefold) isn't implemented yet — only the
  50-move rule is checked. Add a position-history hash count for full draw
  detection.
- **Check evasions in quiescence** aren't handled (quiescence assumes not
  in check) — a common simplification, but can cause some tactical blind
  spots when the side to move is in check at a leaf node.

## Sample output

`dotnet run -c Release -- cli`, bot to move from the starting position:

```
info depth 1 score cp 56 nodes 51 time 5 pv d2d4
info depth 2 score cp 0 nodes 211 time 7 pv d2d4
info depth 3 score cp 50 nodes 567 time 9 pv d2d4
info depth 4 score cp 0 nodes 3050 time 27 pv d2d4
info depth 5 score cp 43 nodes 6067 time 39 pv d2d4
info depth 6 score cp 5 nodes 36060 time 131 pv e2e4
info depth 7 score cp 42 nodes 103641 time 341 pv b1c3
info depth 8 score cp 13 nodes 273008 time 495 pv e2e4
info depth 9 score cp 42 nodes 729014 time 837 pv e2e4
info depth 10 score cp 8 nodes 2327142 time 2314 pv g1f3
info depth 11 score cp 37 nodes 4612155 time 3542 pv g1f3
info depth 12 score cp 8 nodes 18449076 time 11476 pv g1f3
info depth 13 score cp 30 nodes 47404449 time 28044 pv g1f3
Bot plays: g1f3
8 r n b q k b n r
7 p p p p p p p p
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . N . .
2 P P P P P P P P
1 R N B Q K B . R
  a b c d e f g h
rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1
```
