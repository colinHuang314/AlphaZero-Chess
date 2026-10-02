using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace ChessBot
{


public sealed class Search
{
    public const int Infinity = 1_000_000;
    public const int MateScore = 900_000;
    public const int MaxPly = 128;

    private readonly TranspositionTable _tt;
    private readonly Move[,] _killers = new Move[MaxPly, 2];
    private readonly int[,,] _history = new int[2, 64, 64];
    private Stopwatch _timer = new();
    private long _timeLimitMs;
    private bool _stop;
    private long _nodes;

    public Move BestMoveRoot;
    public int BestScoreRoot;

    public Search(TranspositionTable tt)
    {
        _tt = tt;
    }

    public Move FindBestMove(Board b, int maxDepth, long timeLimitMs)
    {
        _timer = Stopwatch.StartNew();
        _timeLimitMs = timeLimitMs;
        _stop = false;
        _nodes = 0;

        Move bestMove = Move.None;
        int bestScore = 0;

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            int score = NegaMax(b, depth, 0, -Infinity, Infinity, true);
            if (_stop && depth > 1) break;

            if (BestMoveRoot != Move.None)
            {
                bestMove = BestMoveRoot;
                bestScore = score;
            }

            Console.WriteLine($"info depth {depth} score cp {score} nodes {_nodes} time {_timer.ElapsedMilliseconds} pv {bestMove}");

            if (_timer.ElapsedMilliseconds > _timeLimitMs) break;
        }

        BestScoreRoot = bestScore;
        return bestMove;
    }

    private bool TimeUp() => (_nodes & 2047) == 0 && _timer.ElapsedMilliseconds > _timeLimitMs;

    private int NegaMax(Board b, int depth, int ply, int alpha, int beta, bool isRoot)
    {
        _nodes++;

        if (!isRoot && TimeUp()) { _stop = true; return 0; }
        if (_stop) return 0;

        // Draw detection: 50-move rule (repetition detection omitted for simplicity)
        if (!isRoot && b.HalfmoveClock >= 100) return 0;

        ulong key = b.Hash;
        Move ttMove = Move.None;
        if (_tt.TryGet(key, out var entry))
        {
            ttMove = entry.BestMove;
            if (!isRoot && entry.Depth >= depth)
            {
                if (entry.Flag == TTFlag.Exact) return entry.Score;
                if (entry.Flag == TTFlag.LowerBound && entry.Score > alpha) alpha = entry.Score;
                else if (entry.Flag == TTFlag.UpperBound && entry.Score < beta) beta = entry.Score;
                if (alpha >= beta) return entry.Score;
            }
        }

        bool inCheck = b.InCheck(b.SideToMove);

        if (depth <= 0)
            return Quiescence(b, alpha, beta, ply);

        var moves = MoveGen.GenerateLegalMoves(b);

        if (moves.Count == 0)
            return inCheck ? -MateScore + ply : 0; // checkmate or stalemate

        OrderMoves(b, moves, ttMove, ply);

        int origAlpha = alpha;
        Move bestMove = moves[0];
        int bestScore = -Infinity;

        for (int i = 0; i < moves.Count; i++)
        {
            var m = moves[i];
            b.MakeMove(m);
            int score;

            if (i == 0)
            {
                score = -NegaMax(b, depth - 1, ply + 1, -beta, -alpha, false);
            }
            else
            {
                // Late move reduction for quiet, later moves
                int reduction = (depth >= 3 && i >= 4 && !m.Flag().IsCapture() && !inCheck) ? 1 : 0;
                score = -NegaMax(b, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha, false);
                if (score > alpha && (reduction > 0 || score < beta))
                    score = -NegaMax(b, depth - 1, ply + 1, -beta, -alpha, false);
            }

            b.UnmakeMove(m);

            if (_stop) return 0;

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = m;
                if (isRoot) { BestMoveRoot = m; }
            }

            if (score > alpha)
            {
                alpha = score;
            }

            if (alpha >= beta)
            {
                if (!m.Flag().IsCapture())
                {
                    _killers[ply, 1] = _killers[ply, 0];
                    _killers[ply, 0] = m;
                    _history[(int)b.SideToMove, m.From(), m.To()] += depth * depth;
                }
                break;
            }
        }

        TTFlag flag = bestScore <= origAlpha ? TTFlag.UpperBound :
                      bestScore >= beta ? TTFlag.LowerBound : TTFlag.Exact;
        _tt.Store(key, depth, bestScore, flag, bestMove);

        return bestScore;
    }

    private int Quiescence(Board b, int alpha, int beta, int ply)
    {
        _nodes++;
        if (TimeUp()) { _stop = true; return 0; }

        int standPat = Evaluation.Evaluate(b);
        if (standPat >= beta) return beta;
        if (standPat > alpha) alpha = standPat;

        var captures = MoveGen.GenerateLegalCaptures(b);
        OrderCaptures(b, captures);

        foreach (var m in captures)
        {
            b.MakeMove(m);
            int score = -Quiescence(b, -beta, -alpha, ply + 1);
            b.UnmakeMove(m);

            if (_stop) return 0;

            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        return alpha;
    }

    private static int PieceValue(Piece p) => p == Piece.None ? 0 : Evaluation.MgValue[p.Kind()];

    private void OrderMoves(Board b, List<Move> moves, Move ttMove, int ply)
    {
        var scores = new int[moves.Count];
        for (int i = 0; i < moves.Count; i++)
        {
            var m = moves[i];
            if (m == ttMove) scores[i] = 1_000_000;
            else if (m.Flag().IsCapture())
                scores[i] = 100_000 + PieceValue(m.Captured()) * 10 - PieceValue(m.Moved());
            else if (m == _killers[ply, 0]) scores[i] = 90_000;
            else if (m == _killers[ply, 1]) scores[i] = 80_000;
            else scores[i] = _history[(int)b.SideToMove, m.From(), m.To()];
        }
        SortByScoreDesc(moves, scores);
    }

    private void OrderCaptures(Board b, List<Move> moves)
    {
        var scores = new int[moves.Count];
        for (int i = 0; i < moves.Count; i++)
        {
            var m = moves[i];
            scores[i] = PieceValue(m.Captured()) * 10 - PieceValue(m.Moved());
        }
        SortByScoreDesc(moves, scores);
    }

    private static void SortByScoreDesc(List<Move> moves, int[] scores)
    {
        // simple insertion sort - move lists are short, and this avoids allocations
        for (int i = 1; i < moves.Count; i++)
        {
            int key = scores[i];
            Move keyMove = moves[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < key)
            {
                scores[j + 1] = scores[j];
                moves[j + 1] = moves[j];
                j--;
            }
            scores[j + 1] = key;
            moves[j + 1] = keyMove;
        }
    }
}

}
