using System;

namespace ChessBot
{
public static class Zobrist
{
    public static readonly ulong[,] PieceSquare = new ulong[12, 64];
    public static readonly ulong[] Castling = new ulong[16]; // castling rights bitmask 0-15
    public static readonly ulong[] EnPassantFile = new ulong[8];
    public static readonly ulong SideToMove;

    static Zobrist()
    {
        var rng = new Random(20240614); // fixed seed for reproducibility
        byte[] buf = new byte[8];

        ulong NextU64()
        {
            rng.NextBytes(buf);
            return BitConverter.ToUInt64(buf, 0);
        }

        for (int p = 0; p < 12; p++)
            for (int s = 0; s < 64; s++)
                PieceSquare[p, s] = NextU64();

        for (int i = 0; i < 16; i++)
            Castling[i] = NextU64();

        for (int i = 0; i < 8; i++)
            EnPassantFile[i] = NextU64();

        SideToMove = NextU64();
    }
}

}
