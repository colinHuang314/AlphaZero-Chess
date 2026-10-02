using System;

namespace ChessBot
{
public static class Evaluation
{
    // Material values: mg/eg, indexed by piece kind (pawn,knight,bishop,rook,queen,king)
    public static readonly int[] MgValue = { 82, 337, 365, 477, 1025, 0 };
    public static readonly int[] EgValue = { 94, 281, 297, 512, 936, 0 };

    // Game phase weight per piece kind, used for tapering (king/pawn excluded from phase calc)
    private static readonly int[] PhaseWeight = { 0, 1, 1, 2, 4, 0 };
    private const int TotalPhase = 24; // 4 knights+4 bishops... actually 2N+2B+2R+1Q per side*2 => 2*1*4+2*2+... computed below matches standard 24

    // PeSTO-style piece-square tables, white's perspective, a1=index0..h8=index63 with rank8 first in source arrays (we'll flip)
    private static readonly int[] PawnMg = {
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0
    };
    private static readonly int[] PawnEg = {
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0
    };
    private static readonly int[] KnightMg = {
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23
    };
    private static readonly int[] KnightEg = {
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64
    };
    private static readonly int[] BishopMg = {
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21
    };
    private static readonly int[] BishopEg = {
        -14, -21, -11,  -8, -7,  -9, -17, -24,
         -8,  -4,   7, -12, -3, -13,  -4, -14,
          2,  -8,   0,  -1, -2,   6,   0,   4,
         -3,   9,  12,   9, 14,  10,   3,   2,
         -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17
    };
    private static readonly int[] RookMg = {
         32,  42,  32,  51, 63,  9,  31,  43,
         27,  32,  58,  62, 80, 67,  26,  44,
         -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26
    };
    private static readonly int[] RookEg = {
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
         7,  7,  7,  5,  4,  -3,  -5,  -3,
         4,  3, 13,  1,  2,   1,  -1,   2,
         3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20
    };
    private static readonly int[] QueenMg = {
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50
    };
    private static readonly int[] QueenEg = {
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41
    };
    private static readonly int[] KingMg = {
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14
    };
    private static readonly int[] KingEg = {
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
    };

    private static readonly int[][] Mg = { PawnMg, KnightMg, BishopMg, RookMg, QueenMg, KingMg };
    private static readonly int[][] Eg = { PawnEg, KnightEg, BishopEg, RookEg, QueenEg, KingEg };

    // Convert table index (defined rank8->rank1 top to bottom, a->h) to our square index (a1=0)
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static int FlipTableIndex(int sq)
    {
        int file = Sq.File(sq), rank = Sq.Rank(sq);
        int tableRankFromTop = 7 - rank;
        return tableRankFromTop * 8 + file;
    }

    public static int Evaluate(Board b)
    {
        int mgScore = 0, egScore = 0, phase = 0;

        for (int kind = 0; kind < 6; kind++)
        {
            Piece wp = (Piece)kind;
            Piece bp = (Piece)(kind + 6);

            ulong wbb = b.PieceBB[(int)wp];
            while (wbb != 0)
            {
                int sq = Bits.PopLsb(ref wbb);
                int idx = FlipTableIndex(sq);
                mgScore += MgValue[kind] + Mg[kind][idx];
                egScore += EgValue[kind] + Eg[kind][idx];
                phase += PhaseWeight[kind];
            }

            ulong bbb = b.PieceBB[(int)bp];
            while (bbb != 0)
            {
                int sq = Bits.PopLsb(ref bbb);
                // mirror square vertically for black, then flip table index the same way
                int mirroredSq = Sq.Make(Sq.File(sq), 7 - Sq.Rank(sq));
                int idx = FlipTableIndex(mirroredSq);
                mgScore -= MgValue[kind] + Mg[kind][idx];
                egScore -= EgValue[kind] + Eg[kind][idx];
                phase += PhaseWeight[kind];
            }
        }

        // Mobility bonus (cheap approximation): count pseudo attack squares for knights/bishops/rooks/queens
        mgScore += MobilityScore(b, Color.White) - MobilityScore(b, Color.Black);
        egScore += (MobilityScore(b, Color.White) - MobilityScore(b, Color.Black)) / 2;

        int clampedPhase = Math.Min(phase, TotalPhase);
        int mgWeight = clampedPhase;
        int egWeight = TotalPhase - clampedPhase;
        int score = (mgScore * mgWeight + egScore * egWeight) / TotalPhase;

        return b.SideToMove == Color.White ? score : -score;
    }

    private static int MobilityScore(Board b, Color side)
    {
        ulong occ = b.Occupancy;
        ulong own = b.ColorBB[(int)side];
        int baseIdx = side == Color.White ? 0 : 6;
        int score = 0;

        ulong knights = b.PieceBB[baseIdx + 1];
        while (knights != 0)
        {
            int sq = Bits.PopLsb(ref knights);
            score += Bits.PopCount(Attacks.Knight[sq] & ~own) * 4;
        }
        ulong bishops = b.PieceBB[baseIdx + 2];
        while (bishops != 0)
        {
            int sq = Bits.PopLsb(ref bishops);
            score += Bits.PopCount(Attacks.Bishop(sq, occ) & ~own) * 3;
        }
        ulong rooks = b.PieceBB[baseIdx + 3];
        while (rooks != 0)
        {
            int sq = Bits.PopLsb(ref rooks);
            score += Bits.PopCount(Attacks.Rook(sq, occ) & ~own) * 2;
        }
        ulong queens = b.PieceBB[baseIdx + 4];
        while (queens != 0)
        {
            int sq = Bits.PopLsb(ref queens);
            score += Bits.PopCount(Attacks.Queen(sq, occ) & ~own) * 1;
        }
        return score;
    }
}

}
