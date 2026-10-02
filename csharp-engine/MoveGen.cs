using System;
using System.Collections.Generic;

namespace ChessBot
{
public static class MoveGen
{
    public static List<Move> GenerateLegalMoves(Board b)
    {
        var pseudo = new List<Move>(64);
        GeneratePseudoLegal(b, pseudo, capturesOnly: false);

        var legal = new List<Move>(pseudo.Count);
        Color us = b.SideToMove;
        foreach (var m in pseudo)
        {
            b.MakeMove(m);
            if (!b.InCheck(us)) legal.Add(m);
            b.UnmakeMove(m);
        }
        return legal;
    }

    public static List<Move> GenerateLegalCaptures(Board b)
    {
        var pseudo = new List<Move>(32);
        GeneratePseudoLegal(b, pseudo, capturesOnly: true);

        var legal = new List<Move>(pseudo.Count);
        Color us = b.SideToMove;
        foreach (var m in pseudo)
        {
            b.MakeMove(m);
            if (!b.InCheck(us)) legal.Add(m);
            b.UnmakeMove(m);
        }
        return legal;
    }

    private static void GeneratePseudoLegal(Board b, List<Move> moves, bool capturesOnly)
    {
        Color us = b.SideToMove;
        Color them = us == Color.White ? Color.Black : Color.White;
        ulong ownOcc = b.ColorBB[(int)us];
        ulong theirOcc = b.ColorBB[(int)them];
        ulong occ = b.Occupancy;

        GeneratePawnMoves(b, moves, us, ownOcc, theirOcc, capturesOnly);

        AddPieceMoves(b, moves, us == Color.White ? Piece.WKnight : Piece.BKnight, ownOcc, theirOcc,
            sq => Attacks.Knight[sq], capturesOnly);

        AddPieceMoves(b, moves, us == Color.White ? Piece.WBishop : Piece.BBishop, ownOcc, theirOcc,
            sq => Attacks.Bishop(sq, occ), capturesOnly);

        AddPieceMoves(b, moves, us == Color.White ? Piece.WRook : Piece.BRook, ownOcc, theirOcc,
            sq => Attacks.Rook(sq, occ), capturesOnly);

        AddPieceMoves(b, moves, us == Color.White ? Piece.WQueen : Piece.BQueen, ownOcc, theirOcc,
            sq => Attacks.Queen(sq, occ), capturesOnly);

        AddPieceMoves(b, moves, us == Color.White ? Piece.WKing : Piece.BKing, ownOcc, theirOcc,
            sq => Attacks.King[sq], capturesOnly);

        if (!capturesOnly)
            GenerateCastling(b, moves, us);
    }

    private static void AddPieceMoves(Board b, List<Move> moves, Piece piece, ulong ownOcc, ulong theirOcc,
        Func<int, ulong> attackFn, bool capturesOnly)
    {
        ulong bb = b.PieceBB[(int)piece];
        while (bb != 0)
        {
            int from = Bits.PopLsb(ref bb);
            ulong attacks = attackFn(from) & ~ownOcc;
            ulong targets = capturesOnly ? attacks & theirOcc : attacks;
            ulong t = targets;
            while (t != 0)
            {
                int to = Bits.PopLsb(ref t);
                Piece captured = b.SquareToPiece[to];
                MoveFlag flag = captured != Piece.None ? MoveFlag.Capture : MoveFlag.Quiet;
                moves.Add(new Move(from, to, flag, piece, captured));
            }
        }
    }

    private static void GeneratePawnMoves(Board b, List<Move> moves, Color us, ulong ownOcc, ulong theirOcc, bool capturesOnly)
    {
        Piece pawn = us == Color.White ? Piece.WPawn : Piece.BPawn;
        ulong occ = b.Occupancy;
        ulong pawns = b.PieceBB[(int)pawn];
        int pushDir = us == Color.White ? 8 : -8;
        ulong promoRank = us == Color.White ? Bits.Rank8 : Bits.Rank1;
        ulong startRank = us == Color.White ? Bits.Rank2 : Bits.Rank7;

        ulong bb = pawns;
        while (bb != 0)
        {
            int from = Bits.PopLsb(ref bb);
            ulong fromBit = Bits.Bit(from);
            int to1 = from + pushDir;

            if (!capturesOnly && to1 is >= 0 and < 64 && (occ & Bits.Bit(to1)) == 0)
            {
                if ((Bits.Bit(to1) & promoRank) != 0)
                {
                    AddPromotions(moves, from, to1, pawn, Piece.None, false);
                }
                else
                {
                    moves.Add(new Move(from, to1, MoveFlag.Quiet, pawn));
                    if ((fromBit & startRank) != 0)
                    {
                        int to2 = from + pushDir * 2;
                        if ((occ & Bits.Bit(to2)) == 0)
                            moves.Add(new Move(from, to2, MoveFlag.DoublePawnPush, pawn));
                    }
                }
            }

            // Captures
            ulong attacks = Attacks.Pawn[(int)us, from] & theirOcc;
            ulong t = attacks;
            while (t != 0)
            {
                int to = Bits.PopLsb(ref t);
                Piece captured = b.SquareToPiece[to];
                if ((Bits.Bit(to) & promoRank) != 0)
                    AddPromotions(moves, from, to, pawn, captured, true);
                else
                    moves.Add(new Move(from, to, MoveFlag.Capture, pawn, captured));
            }

            // En passant
            if (b.EnPassantSquare >= 0)
            {
                ulong epAttack = Attacks.Pawn[(int)us, from] & Bits.Bit(b.EnPassantSquare);
                if (epAttack != 0)
                {
                    Piece capturedPawn = us == Color.White ? Piece.BPawn : Piece.WPawn;
                    moves.Add(new Move(from, b.EnPassantSquare, MoveFlag.EnPassant, pawn, capturedPawn));
                }
            }
        }
    }

    private static void AddPromotions(List<Move> moves, int from, int to, Piece pawn, Piece captured, bool isCapture)
    {
        if (isCapture)
        {
            moves.Add(new Move(from, to, MoveFlag.PromoCaptureQueen, pawn, captured));
            moves.Add(new Move(from, to, MoveFlag.PromoCaptureRook, pawn, captured));
            moves.Add(new Move(from, to, MoveFlag.PromoCaptureBishop, pawn, captured));
            moves.Add(new Move(from, to, MoveFlag.PromoCaptureKnight, pawn, captured));
        }
        else
        {
            moves.Add(new Move(from, to, MoveFlag.PromoQueen, pawn));
            moves.Add(new Move(from, to, MoveFlag.PromoRook, pawn));
            moves.Add(new Move(from, to, MoveFlag.PromoBishop, pawn));
            moves.Add(new Move(from, to, MoveFlag.PromoKnight, pawn));
        }
    }

    private static void GenerateCastling(Board b, List<Move> moves, Color us)
    {
        ulong occ = b.Occupancy;
        Color them = us == Color.White ? Color.Black : Color.White;

        if (us == Color.White)
        {
            if ((b.CastlingRights & Castle.WK) != 0 &&
                (occ & (Bits.Bit(Sq.F1) | Bits.Bit(Sq.G1))) == 0 &&
                !b.IsSquareAttacked(Sq.E1, them) && !b.IsSquareAttacked(Sq.F1, them) && !b.IsSquareAttacked(Sq.G1, them))
            {
                moves.Add(new Move(Sq.E1, Sq.G1, MoveFlag.KingCastle, Piece.WKing));
            }
            if ((b.CastlingRights & Castle.WQ) != 0 &&
                (occ & (Bits.Bit(Sq.B1) | Bits.Bit(Sq.C1) | Bits.Bit(Sq.D1))) == 0 &&
                !b.IsSquareAttacked(Sq.E1, them) && !b.IsSquareAttacked(Sq.D1, them) && !b.IsSquareAttacked(Sq.C1, them))
            {
                moves.Add(new Move(Sq.E1, Sq.C1, MoveFlag.QueenCastle, Piece.WKing));
            }
        }
        else
        {
            if ((b.CastlingRights & Castle.BK) != 0 &&
                (occ & (Bits.Bit(Sq.F8) | Bits.Bit(Sq.G8))) == 0 &&
                !b.IsSquareAttacked(Sq.E8, them) && !b.IsSquareAttacked(Sq.F8, them) && !b.IsSquareAttacked(Sq.G8, them))
            {
                moves.Add(new Move(Sq.E8, Sq.G8, MoveFlag.KingCastle, Piece.BKing));
            }
            if ((b.CastlingRights & Castle.BQ) != 0 &&
                (occ & (Bits.Bit(Sq.B8) | Bits.Bit(Sq.C8) | Bits.Bit(Sq.D8))) == 0 &&
                !b.IsSquareAttacked(Sq.E8, them) && !b.IsSquareAttacked(Sq.D8, them) && !b.IsSquareAttacked(Sq.C8, them))
            {
                moves.Add(new Move(Sq.E8, Sq.C8, MoveFlag.QueenCastle, Piece.BKing));
            }
        }
    }
}

}
