using System;
using System.Runtime.CompilerServices;

namespace ChessBot
{

public enum Dir { N = 0, S, E, W, NE, NW, SE, SW }

public static class Bits
{
    public const ulong FileA = 0x0101010101010101UL;
    public const ulong FileH = 0x8080808080808080UL;
    public const ulong Rank1 = 0x00000000000000FFUL;
    public const ulong Rank2 = 0x000000000000FF00UL;
    public const ulong Rank4 = 0x00000000FF000000UL;
    public const ulong Rank5 = 0x000000FF00000000UL;
    public const ulong Rank7 = 0x00FF000000000000UL;
    public const ulong Rank8 = 0xFF00000000000000UL;
    public const ulong NotFileA = ~FileA;
    public const ulong NotFileH = ~FileH;

    // Manual bit-twiddling implementations (no System.Numerics.BitOperations dependency,
    // since that can be inaccessible under some Unity API compatibility levels).

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PopCount(ulong x)
    {
        x -= (x >> 1) & 0x5555555555555555UL;
        x = (x & 0x3333333333333333UL) + ((x >> 2) & 0x3333333333333333UL);
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fUL;
        return (int)((x * 0x0101010101010101UL) >> 56);
    }

    // Index of the highest set bit. Undefined for b == 0 (never called that way here).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Msb(ulong b)
    {
        b |= b >> 1;
        b |= b >> 2;
        b |= b >> 4;
        b |= b >> 8;
        b |= b >> 16;
        b |= b >> 32;
        return PopCount(b) - 1;
    }

    // Index of the lowest set bit. Undefined for b == 0 (never called that way here).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Lsb(ulong b)
    {
        ulong isolated = b & (ulong)(-(long)b);
        return Msb(isolated);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PopLsb(ref ulong b)
    {
        int i = Lsb(b);
        b &= b - 1;
        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Bit(int sq) => 1UL << sq;
}

public static class Attacks
{
    public static readonly ulong[] Knight = new ulong[64];
    public static readonly ulong[] King = new ulong[64];
    public static readonly ulong[,] Pawn = new ulong[2, 64]; // [color, sq]

    // Rays FROM sq in each direction, exclusive of sq, up to (not including) wrap
    public static readonly ulong[,] Rays = new ulong[8, 64];

    static Attacks()
    {
        InitLeapers();
        InitRays();
    }

    private static void InitLeapers()
    {
        for (int sq = 0; sq < 64; sq++)
        {
            int f = Sq.File(sq), r = Sq.Rank(sq);
            ulong k = 0, n = 0;

            // King: 8 neighbors
            for (int df = -1; df <= 1; df++)
            for (int dr = -1; dr <= 1; dr++)
            {
                if (df == 0 && dr == 0) continue;
                int nf = f + df, nr = r + dr;
                if (nf is >= 0 and < 8 && nr is >= 0 and < 8)
                    k |= Bits.Bit(Sq.Make(nf, nr));
            }
            King[sq] = k;

            // Knight: 8 L-shapes
            (int, int)[] offsets = { (1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2) };
            foreach (var (df, dr) in offsets)
            {
                int nf = f + df, nr = r + dr;
                if (nf is >= 0 and < 8 && nr is >= 0 and < 8)
                    n |= Bits.Bit(Sq.Make(nf, nr));
            }
            Knight[sq] = n;

            // Pawn attacks
            ulong wp = 0, bp = 0;
            if (r < 7)
            {
                if (f > 0) wp |= Bits.Bit(Sq.Make(f - 1, r + 1));
                if (f < 7) wp |= Bits.Bit(Sq.Make(f + 1, r + 1));
            }
            if (r > 0)
            {
                if (f > 0) bp |= Bits.Bit(Sq.Make(f - 1, r - 1));
                if (f < 7) bp |= Bits.Bit(Sq.Make(f + 1, r - 1));
            }
            Pawn[(int)Color.White, sq] = wp;
            Pawn[(int)Color.Black, sq] = bp;
        }
    }

    private static void InitRays()
    {
        (int df, int dr)[] dirs =
        {
            (0,1), (0,-1), (1,0), (-1,0), // N, S, E, W
            (1,1), (-1,1), (1,-1), (-1,-1) // NE, NW, SE, SW
        };

        for (int d = 0; d < 8; d++)
        {
            var (df, dr) = dirs[d];
            for (int sq = 0; sq < 64; sq++)
            {
                int f = Sq.File(sq), r = Sq.Rank(sq);
                ulong ray = 0;
                int nf = f + df, nr = r + dr;
                while (nf is >= 0 and < 8 && nr is >= 0 and < 8)
                {
                    ray |= Bits.Bit(Sq.Make(nf, nr));
                    nf += df; nr += dr;
                }
                Rays[d, sq] = ray;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Bishop(int sq, ulong occupancy)
    {
        int file = Sq.File(sq);
        int rank = Sq.Rank(sq);
        ulong attacks = 0;

        for (int f = file + 1, r = rank + 1; f < 8 && r < 8; f++, r++)
        {
            int ns = Sq.Make(f, r);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }
        for (int f = file - 1, r = rank + 1; f >= 0 && r < 8; f--, r++)
        {
            int ns = Sq.Make(f, r);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }
        for (int f = file + 1, r = rank - 1; f < 8 && r >= 0; f++, r--)
        {
            int ns = Sq.Make(f, r);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }
        for (int f = file - 1, r = rank - 1; f >= 0 && r >= 0; f--, r--)
        {
            int ns = Sq.Make(f, r);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }

        return attacks;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Rook(int sq, ulong occupancy)
    {
        int file = Sq.File(sq);
        int rank = Sq.Rank(sq);
        ulong attacks = 0;

        for (int r = rank + 1; r < 8; r++)
        {
            int ns = Sq.Make(file, r);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }
        for (int r = rank - 1; r >= 0; r--)
        {
            int ns = Sq.Make(file, r);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }
        for (int f = file + 1; f < 8; f++)
        {
            int ns = Sq.Make(f, rank);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }
        for (int f = file - 1; f >= 0; f--)
        {
            int ns = Sq.Make(f, rank);
            attacks |= Bits.Bit(ns);
            if (((occupancy >> ns) & 1UL) != 0) break;
        }

        return attacks;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Queen(int sq, ulong occupancy) => Bishop(sq, occupancy) | Rook(sq, occupancy);
}

}
