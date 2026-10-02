using System;
using System.Runtime.CompilerServices;

namespace ChessBot
{


// Packed as: bits 0-5 = from, 6-11 = to, 12-15 = flag, 16-19 = moved piece, 20-23 = captured piece
public readonly struct Move : IEquatable<Move>
{
    public readonly int Data;

    public static readonly Move None = new Move(0);

    private Move(int data) => Data = data;

    public Move(int from, int to, MoveFlag flag, Piece moved, Piece captured = Piece.None)
    {
        Data = (from & 0x3F) | ((to & 0x3F) << 6) | (((int)flag & 0xF) << 12)
             | (((int)moved & 0xF) << 16) | (((int)captured & 0xF) << 20);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int From() => Data & 0x3F;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int To() => (Data >> 6) & 0x3F;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MoveFlag Flag() => (MoveFlag)((Data >> 12) & 0xF);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Piece Moved() => (Piece)((Data >> 16) & 0xF);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Piece Captured() => (Piece)((Data >> 20) & 0xF);

    public bool IsNull => Data == 0 && Moved() == Piece.None;

    public bool Equals(Move other) => Data == other.Data;
    public override bool Equals(object? obj) => obj is Move m && Equals(m);
    public override int GetHashCode() => Data;
    public static bool operator ==(Move a, Move b) => a.Data == b.Data;
    public static bool operator !=(Move a, Move b) => a.Data != b.Data;

    public override string ToString()
    {
        string s = Sq.Name(From()) + Sq.Name(To());
        var f = Flag();
        if (f.IsPromotion())
        {
            char c = f switch
            {
                MoveFlag.PromoKnight or MoveFlag.PromoCaptureKnight => 'n',
                MoveFlag.PromoBishop or MoveFlag.PromoCaptureBishop => 'b',
                MoveFlag.PromoRook or MoveFlag.PromoCaptureRook => 'r',
                _ => 'q'
            };
            s += c;
        }
        return s;
    }
}

}
