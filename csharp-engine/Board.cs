using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace ChessBot
{


// Castling rights bits
public static class Castle
{
    public const int WK = 1, WQ = 2, BK = 4, BQ = 8;
}

public struct UndoInfo
{
    public Piece Captured;
    public int CastlingRights;
    public int EnPassantSquare;
    public int HalfmoveClock;
    public ulong Hash;
}

public sealed class Board
{
    public readonly ulong[] PieceBB = new ulong[12];
    public readonly ulong[] ColorBB = new ulong[2];
    public ulong Occupancy;
    public readonly Piece[] SquareToPiece = new Piece[64];
    public Color SideToMove;
    public int CastlingRights;
    public int EnPassantSquare; // -1 if none
    public int HalfmoveClock;
    public int FullmoveNumber;
    public ulong Hash;

    private readonly List<UndoInfo> _history = new(256);
    private readonly List<Move> _moveHistory = new(256);

    public const string StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    public Board()
    {
        SetFen(StartFen);
    }

    public void SetFen(string fen)
    {
        Array.Fill(SquareToPiece, Piece.None);
        Array.Clear(PieceBB);
        ColorBB[0] = ColorBB[1] = 0;
        Occupancy = 0;
        _history.Clear();
        _moveHistory.Clear();

        var parts = fen.Split(' ');
        string boardPart = parts[0];
        int rank = 7, file = 0;
        foreach (char c in boardPart)
        {
            if (c == '/') { rank--; file = 0; }
            else if (char.IsDigit(c)) { file += c - '0'; }
            else
            {
                Piece p = CharToPiece(c);
                int sq = Sq.Make(file, rank);
                PlacePiece(p, sq);
                file++;
            }
        }

        SideToMove = parts.Length > 1 && parts[1] == "b" ? Color.Black : Color.White;

        CastlingRights = 0;
        if (parts.Length > 2)
        {
            string cr = parts[2];
            if (cr.Contains('K')) CastlingRights |= Castle.WK;
            if (cr.Contains('Q')) CastlingRights |= Castle.WQ;
            if (cr.Contains('k')) CastlingRights |= Castle.BK;
            if (cr.Contains('q')) CastlingRights |= Castle.BQ;
        }

        EnPassantSquare = -1;
        if (parts.Length > 3 && parts[3] != "-")
            EnPassantSquare = Sq.Parse(parts[3]);

        HalfmoveClock = parts.Length > 4 ? int.Parse(parts[4]) : 0;
        FullmoveNumber = parts.Length > 5 ? int.Parse(parts[5]) : 1;

        Hash = ComputeHash();
    }

    private static Piece CharToPiece(char c) => c switch
    {
        'P' => Piece.WPawn, 'N' => Piece.WKnight, 'B' => Piece.WBishop,
        'R' => Piece.WRook, 'Q' => Piece.WQueen, 'K' => Piece.WKing,
        'p' => Piece.BPawn, 'n' => Piece.BKnight, 'b' => Piece.BBishop,
        'r' => Piece.BRook, 'q' => Piece.BQueen, 'k' => Piece.BKing,
        _ => Piece.None
    };

    public ulong ComputeHash()
    {
        ulong h = 0;
        for (int sq = 0; sq < 64; sq++)
        {
            Piece p = SquareToPiece[sq];
            if (p != Piece.None) h ^= Zobrist.PieceSquare[(int)p, sq];
        }
        h ^= Zobrist.Castling[CastlingRights];
        if (EnPassantSquare >= 0) h ^= Zobrist.EnPassantFile[Sq.File(EnPassantSquare)];
        if (SideToMove == Color.Black) h ^= Zobrist.SideToMove;
        return h;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void PlacePiece(Piece p, int sq)
    {
        SquareToPiece[sq] = p;
        ulong bit = Bits.Bit(sq);
        PieceBB[(int)p] |= bit;
        ColorBB[(int)p.ColorOf()] |= bit;
        Occupancy |= bit;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void RemovePiece(Piece p, int sq)
    {
        SquareToPiece[sq] = Piece.None;
        ulong bit = ~Bits.Bit(sq);
        PieceBB[(int)p] &= bit;
        ColorBB[(int)p.ColorOf()] &= bit;
        Occupancy &= bit;
    }

    public bool IsSquareAttacked(int sq, Color bySide)
    {
        if (sq < 0 || sq >= 64)
            return false;

        int c = (int)bySide;

        if ((Attacks.Pawn[(int)(bySide == Color.White ? Color.Black : Color.White), sq] &
             PieceBB[bySide == Color.White ? (int)Piece.WPawn : (int)Piece.BPawn]) != 0)
            return true;

        if ((Attacks.Knight[sq] & PieceBB[bySide == Color.White ? (int)Piece.WKnight : (int)Piece.BKnight]) != 0)
            return true;

        if ((Attacks.King[sq] & PieceBB[bySide == Color.White ? (int)Piece.WKing : (int)Piece.BKing]) != 0)
            return true;

        ulong bishopsQueens = PieceBB[bySide == Color.White ? (int)Piece.WBishop : (int)Piece.BBishop] |
                              PieceBB[bySide == Color.White ? (int)Piece.WQueen : (int)Piece.BQueen];
        if ((Attacks.Bishop(sq, Occupancy) & bishopsQueens) != 0) return true;

        ulong rooksQueens = PieceBB[bySide == Color.White ? (int)Piece.WRook : (int)Piece.BRook] |
                            PieceBB[bySide == Color.White ? (int)Piece.WQueen : (int)Piece.BQueen];
        if ((Attacks.Rook(sq, Occupancy) & rooksQueens) != 0) return true;

        return false;
    }

    public int KingSquare(Color side)
    {
        ulong kingBb = PieceBB[side == Color.White ? (int)Piece.WKing : (int)Piece.BKing];
        return kingBb == 0 ? -1 : Bits.Lsb(kingBb);
    }

    public bool InCheck(Color side)
    {
        int kingSq = KingSquare(side);
        if (kingSq < 0) return false;
        return IsSquareAttacked(kingSq, side == Color.White ? Color.Black : Color.White);
    }

    public void MakeMove(Move m)
    {
        var undo = new UndoInfo
        {
            Captured = m.Captured(),
            CastlingRights = CastlingRights,
            EnPassantSquare = EnPassantSquare,
            HalfmoveClock = HalfmoveClock,
            Hash = Hash
        };
        _history.Add(undo);
        _moveHistory.Add(m);

        int from = m.From(), to = m.To();
        Piece moved = m.Moved();
        Piece captured = m.Captured();
        MoveFlag flag = m.Flag();
        Color us = SideToMove;
        Color them = us == Color.White ? Color.Black : Color.White;

        // Remove old EP/castling hash contributions (will be re-added below as needed)
        if (EnPassantSquare >= 0) Hash ^= Zobrist.EnPassantFile[Sq.File(EnPassantSquare)];
        Hash ^= Zobrist.Castling[CastlingRights];

        int newEnPassant = -1;

        // Handle capture (except en passant handled specially)
        if (flag == MoveFlag.EnPassant)
        {
            int capSq = us == Color.White ? to - 8 : to + 8;
            Piece capPawn = us == Color.White ? Piece.BPawn : Piece.WPawn;
            RemovePiece(capPawn, capSq);
            Hash ^= Zobrist.PieceSquare[(int)capPawn, capSq];
        }
        else if (captured != Piece.None)
        {
            RemovePiece(captured, to);
            Hash ^= Zobrist.PieceSquare[(int)captured, to];
        }

        // Move the piece
        RemovePiece(moved, from);
        Hash ^= Zobrist.PieceSquare[(int)moved, from];

        Piece finalPiece = moved;
        if (flag.IsPromotion())
        {
            finalPiece = flag.PromotionKind(us);
        }
        PlacePiece(finalPiece, to);
        Hash ^= Zobrist.PieceSquare[(int)finalPiece, to];

        // Castling rook move
        if (flag == MoveFlag.KingCastle)
        {
            if (us == Color.White)
            {
                RemovePiece(Piece.WRook, Sq.H1);
                PlacePiece(Piece.WRook, Sq.F1);
                Hash ^= Zobrist.PieceSquare[(int)Piece.WRook, Sq.H1];
                Hash ^= Zobrist.PieceSquare[(int)Piece.WRook, Sq.F1];
            }
            else
            {
                RemovePiece(Piece.BRook, Sq.H8);
                PlacePiece(Piece.BRook, Sq.F8);
                Hash ^= Zobrist.PieceSquare[(int)Piece.BRook, Sq.H8];
                Hash ^= Zobrist.PieceSquare[(int)Piece.BRook, Sq.F8];
            }
        }
        else if (flag == MoveFlag.QueenCastle)
        {
            if (us == Color.White)
            {
                RemovePiece(Piece.WRook, Sq.A1);
                PlacePiece(Piece.WRook, Sq.D1);
                Hash ^= Zobrist.PieceSquare[(int)Piece.WRook, Sq.A1];
                Hash ^= Zobrist.PieceSquare[(int)Piece.WRook, Sq.D1];
            }
            else
            {
                RemovePiece(Piece.BRook, Sq.A8);
                PlacePiece(Piece.BRook, Sq.D8);
                Hash ^= Zobrist.PieceSquare[(int)Piece.BRook, Sq.A8];
                Hash ^= Zobrist.PieceSquare[(int)Piece.BRook, Sq.D8];
            }
        }

        // Double pawn push sets EP square
        if (flag == MoveFlag.DoublePawnPush)
        {
            newEnPassant = us == Color.White ? from + 8 : from - 8;
        }

        // Update castling rights
        UpdateCastlingRightsOnMove(from, to, moved);

        EnPassantSquare = newEnPassant;
        if (EnPassantSquare >= 0) Hash ^= Zobrist.EnPassantFile[Sq.File(EnPassantSquare)];
        Hash ^= Zobrist.Castling[CastlingRights];

        // Halfmove clock
        if (moved.Kind() == 0 || captured != Piece.None || flag == MoveFlag.EnPassant)
            HalfmoveClock = 0;
        else
            HalfmoveClock++;

        if (us == Color.Black) FullmoveNumber++;

        SideToMove = them;
        Hash ^= Zobrist.SideToMove;
    }

    private void UpdateCastlingRightsOnMove(int from, int to, Piece moved)
    {
        if (moved == Piece.WKing) CastlingRights &= ~(Castle.WK | Castle.WQ);
        else if (moved == Piece.BKing) CastlingRights &= ~(Castle.BK | Castle.BQ);

        if (from == Sq.A1 || to == Sq.A1) CastlingRights &= ~Castle.WQ;
        if (from == Sq.H1 || to == Sq.H1) CastlingRights &= ~Castle.WK;
        if (from == Sq.A8 || to == Sq.A8) CastlingRights &= ~Castle.BQ;
        if (from == Sq.H8 || to == Sq.H8) CastlingRights &= ~Castle.BK;
    }

    public void UnmakeMove(Move m)
    {
        var undo = _history[^1];
        _history.RemoveAt(_history.Count - 1);
        _moveHistory.RemoveAt(_moveHistory.Count - 1);

        Color them = SideToMove;
        Color us = them == Color.White ? Color.Black : Color.White;
        SideToMove = us;
        if (us == Color.Black) FullmoveNumber--;

        int from = m.From(), to = m.To();
        Piece moved = m.Moved();
        MoveFlag flag = m.Flag();

        Piece finalPiece = flag.IsPromotion() ? flag.PromotionKind(us) : moved;

        RemovePiece(finalPiece, to);
        PlacePiece(moved, from);

        if (flag == MoveFlag.EnPassant)
        {
            int capSq = us == Color.White ? to - 8 : to + 8;
            Piece capPawn = us == Color.White ? Piece.BPawn : Piece.WPawn;
            PlacePiece(capPawn, capSq);
        }
        else if (undo.Captured != Piece.None)
        {
            PlacePiece(undo.Captured, to);
        }

        if (flag == MoveFlag.KingCastle)
        {
            if (us == Color.White) { RemovePiece(Piece.WRook, Sq.F1); PlacePiece(Piece.WRook, Sq.H1); }
            else { RemovePiece(Piece.BRook, Sq.F8); PlacePiece(Piece.BRook, Sq.H8); }
        }
        else if (flag == MoveFlag.QueenCastle)
        {
            if (us == Color.White) { RemovePiece(Piece.WRook, Sq.D1); PlacePiece(Piece.WRook, Sq.A1); }
            else { RemovePiece(Piece.BRook, Sq.D8); PlacePiece(Piece.BRook, Sq.A8); }
        }

        CastlingRights = undo.CastlingRights;
        EnPassantSquare = undo.EnPassantSquare;
        HalfmoveClock = undo.HalfmoveClock;
        Hash = undo.Hash;
    }

    // Null move for null-move pruning
    private readonly Stack<(int ep, ulong hash)> _nullHistory = new();

    public void MakeNullMove()
    {
        _nullHistory.Push((EnPassantSquare, Hash));
        if (EnPassantSquare >= 0) Hash ^= Zobrist.EnPassantFile[Sq.File(EnPassantSquare)];
        EnPassantSquare = -1;
        SideToMove = SideToMove == Color.White ? Color.Black : Color.White;
        Hash ^= Zobrist.SideToMove;
    }

    public void UnmakeNullMove()
    {
        var (ep, hash) = _nullHistory.Pop();
        EnPassantSquare = ep;
        SideToMove = SideToMove == Color.White ? Color.Black : Color.White;
        Hash = hash;
    }

    public string ToFen()
    {
        var sb = new StringBuilder();
        for (int rank = 7; rank >= 0; rank--)
        {
            int empty = 0;
            for (int file = 0; file < 8; file++)
            {
                Piece p = SquareToPiece[Sq.Make(file, rank)];
                if (p == Piece.None) { empty++; continue; }
                if (empty > 0) { sb.Append(empty); empty = 0; }
                sb.Append(p.ToChar());
            }
            if (empty > 0) sb.Append(empty);
            if (rank > 0) sb.Append('/');
        }
        sb.Append(SideToMove == Color.White ? " w " : " b ");
        string cr = "";
        if ((CastlingRights & Castle.WK) != 0) cr += "K";
        if ((CastlingRights & Castle.WQ) != 0) cr += "Q";
        if ((CastlingRights & Castle.BK) != 0) cr += "k";
        if ((CastlingRights & Castle.BQ) != 0) cr += "q";
        sb.Append(cr == "" ? "-" : cr);
        sb.Append(' ');
        sb.Append(EnPassantSquare >= 0 ? Sq.Name(EnPassantSquare) : "-");
        sb.Append(' ').Append(HalfmoveClock).Append(' ').Append(FullmoveNumber);
        return sb.ToString();
    }

    public string ToPgn()
    {
        var sb = new StringBuilder();
        sb.AppendLine("[Event \"ChessBot game\"]");
        sb.AppendLine("[Site \"Local\"]");
        sb.AppendLine("[Date \"2026.07.14\"]");
        sb.AppendLine("[Round \"-\"]");
        sb.AppendLine("[White \"Human\"]");
        sb.AppendLine("[Black \"ChessBot\"]");
        sb.AppendLine("[Result \"*\"]");
        sb.AppendLine();

        if (_moveHistory.Count == 0)
        {
            sb.Append('*');
            return sb.ToString();
        }

        var board = new Board();
        for (int i = 0; i < _moveHistory.Count; i++)
        {
            if (i % 2 == 0)
            {
                if (i > 0) sb.Append(' ');
                sb.Append((i / 2) + 1).Append(". ");
            }
            else
            {
                sb.Append(' ');
            }

            string san = MoveToSan(board, _moveHistory[i]);
            sb.Append(san);
            board.MakeMove(_moveHistory[i]);
        }

        sb.Append(" *");
        return sb.ToString();
    }

    private static string MoveToSan(Board board, Move move)
    {
        var moved = move.Moved();
        var flag = move.Flag();

        if (flag == MoveFlag.KingCastle) return "O-O";
        if (flag == MoveFlag.QueenCastle) return "O-O-O";

        string san;
        if (moved.Kind() == 0)
        {
            san = flag.IsCapture() ? $"{(char)('a' + Sq.File(move.From()))}x{Sq.Name(move.To())}" : Sq.Name(move.To());
        }
        else
        {
            char pieceChar = moved.Kind() switch
            {
                1 => 'N',
                2 => 'B',
                3 => 'R',
                4 => 'Q',
                _ => 'K'
            };

            var legalMoves = MoveGen.GenerateLegalMoves(board);
            bool needsDisambiguation = false;
            foreach (var legal in legalMoves)
            {
                if (legal.To() != move.To()) continue;
                if (legal.Moved().Kind() != moved.Kind()) continue;
                if (legal.From() == move.From()) continue;
                needsDisambiguation = true;
                break;
            }

            san = pieceChar.ToString();
            if (needsDisambiguation)
            {
                bool sameFile = false;
                bool sameRank = false;
                foreach (var legal in legalMoves)
                {
                    if (legal.To() != move.To()) continue;
                    if (legal.Moved().Kind() != moved.Kind()) continue;
                    if (legal.From() == move.From()) continue;
                    if (Sq.File(legal.From()) == Sq.File(move.From())) sameFile = true;
                    if (Sq.Rank(legal.From()) == Sq.Rank(move.From())) sameRank = true;
                }

                if (sameFile && sameRank)
                    san += Sq.Name(move.From());
                else if (sameFile)
                    san += (char)('a' + Sq.File(move.From()));
                else if (sameRank)
                    san += (char)('1' + Sq.Rank(move.From()));
            }

            if (flag.IsCapture()) san += 'x';
            san += Sq.Name(move.To());
        }

        if (flag.IsPromotion())
        {
            char promoChar = flag switch
            {
                MoveFlag.PromoKnight or MoveFlag.PromoCaptureKnight => 'n',
                MoveFlag.PromoBishop or MoveFlag.PromoCaptureBishop => 'b',
                MoveFlag.PromoRook or MoveFlag.PromoCaptureRook => 'r',
                _ => 'q'
            };
            san += '=' + promoChar;
        }

        board.MakeMove(move);
        var opponent = board.SideToMove == Color.White ? Color.Black : Color.White;
        bool isCheck = board.InCheck(opponent);
        var nextMoves = MoveGen.GenerateLegalMoves(board);
        bool isMate = isCheck && nextMoves.Count == 0;
        board.UnmakeMove(move);

        if (isCheck) san += isMate ? '#' : '+';
        return san;
    }

    public void Print()
    {
        for (int rank = 7; rank >= 0; rank--)
        {
            Console.Write(rank + 1);
            Console.Write(" ");
            for (int file = 0; file < 8; file++)
            {
                Console.Write(SquareToPiece[Sq.Make(file, rank)].ToChar());
                Console.Write(" ");
            }
            Console.WriteLine();
        }
        Console.WriteLine("  a b c d e f g h");
    }
}

}
