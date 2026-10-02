using System;

namespace ChessBot
{
public enum TTFlag : byte { Exact, LowerBound, UpperBound }

public struct TTEntry
{
    public ulong Key;
    public int Depth;
    public int Score;
    public TTFlag Flag;
    public Move BestMove;
}

public sealed class TranspositionTable
{
    private readonly TTEntry[] _table;
    private readonly int _mask;

    public TranspositionTable(int sizeMb = 128)
    {
        int entrySize = 32; // approx bytes per entry
        int count = (sizeMb * 1024 * 1024) / entrySize;
        int pow2 = 1;
        while (pow2 * 2 <= count) pow2 *= 2;
        _table = new TTEntry[pow2];
        _mask = pow2 - 1;
    }

    public void Clear() => Array.Clear(_table);

    public bool TryGet(ulong key, out TTEntry entry)
    {
        entry = _table[key & (ulong)_mask];
        return entry.Key == key;
    }

    public void Store(ulong key, int depth, int score, TTFlag flag, Move bestMove)
    {
        int idx = (int)(key & (ulong)_mask);
        ref TTEntry slot = ref _table[idx];
        // Replacement strategy: prefer deeper searches, but always replace if the slot is a different position
        if (slot.Key != key || depth >= slot.Depth)
        {
            slot.Key = key;
            slot.Depth = depth;
            slot.Score = score;
            slot.Flag = flag;
            slot.BestMove = bestMove;
        }
    }
}

}
