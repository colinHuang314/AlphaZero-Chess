using System;
using System.Runtime.CompilerServices;

namespace ChessBot
{


// Squares 0..63, a1=0, b1=1, ... h1=7, a2=8, ... h8=63
public static class Sq
{
    public const int A1=0,B1=1,C1=2,D1=3,E1=4,F1=5,G1=6,H1=7;
    public const int A8=56,B8=57,C8=58,D8=59,E8=60,F8=61,G8=62,H8=63;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int File(int sq) => sq & 7;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Rank(int sq) => sq >> 3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Make(int file, int rank) => rank * 8 + file;

    public static string Name(int sq) => $"{(char)('a' + File(sq))}{Rank(sq) + 1}";

    public static int Parse(string s)
    {
        int file = s[0] - 'a';
        int rank = s[1] - '1';
        return Make(file, rank);
    }
}

public enum Color { White = 0, Black = 1 }

public enum Piece
{
    WPawn = 0, WKnight, WBishop, WRook, WQueen, WKing,
    BPawn, BKnight, BBishop, BRook, BQueen, BKing,
    None
}

public static class PieceEx
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Color ColorOf(this Piece p) => (int)p < 6 ? Color.White : Color.Black;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsWhite(this Piece p) => (int)p < 6;

    // 0..5 regardless of color: pawn,knight,bishop,rook,queen,king
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Kind(this Piece p) => (int)p % 6;

    public static char ToChar(this Piece p)
    {
        char[] chars = { 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k' };
        if (p == Piece.None) return '.';
        return chars[(int)p];
    }
}

public enum MoveFlag
{
    Quiet = 0,
    DoublePawnPush,
    KingCastle,
    QueenCastle,
    Capture,
    EnPassant,
    PromoKnight = 8,
    PromoBishop,
    PromoRook,
    PromoQueen,
    PromoCaptureKnight,
    PromoCaptureBishop,
    PromoCaptureRook,
    PromoCaptureQueen
}

public static class MoveFlagEx
{
    public static bool IsCapture(this MoveFlag f) =>
        f == MoveFlag.Capture || f == MoveFlag.EnPassant ||
        (f >= MoveFlag.PromoCaptureKnight && f <= MoveFlag.PromoCaptureQueen);

    public static bool IsPromotion(this MoveFlag f) => (int)f >= 8;

    public static Piece PromotionKind(this MoveFlag f, Color side)
    {
        int rel = (int)f & 3; // 0=N,1=B,2=R,3=Q for both promo & promo-capture ranges
        int baseIdx = side == Color.White ? 0 : 6;
        return f switch
        {
            MoveFlag.PromoKnight or MoveFlag.PromoCaptureKnight => (Piece)(baseIdx + 1),
            MoveFlag.PromoBishop or MoveFlag.PromoCaptureBishop => (Piece)(baseIdx + 2),
            MoveFlag.PromoRook or MoveFlag.PromoCaptureRook => (Piece)(baseIdx + 3),
            MoveFlag.PromoQueen or MoveFlag.PromoCaptureQueen => (Piece)(baseIdx + 4),
            _ => Piece.None
        };
    }
}

}
