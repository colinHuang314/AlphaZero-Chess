using System;
using System.Collections.Generic;

namespace ChessBot
{
public sealed class Uci
{
    private readonly Board _board = new();
    private readonly TranspositionTable _tt = new(128);

    public void Run()
    {
        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (line.Length == 0) continue;

            if (line == "uci")
            {
                Console.WriteLine("id name ChessBot");
                Console.WriteLine("id author You");
                Console.WriteLine("uciok");
            }
            else if (line == "isready")
            {
                Console.WriteLine("readyok");
            }
            else if (line == "ucinewgame")
            {
                _board.SetFen(Board.StartFen);
                _tt.Clear();
            }
            else if (line.StartsWith("position"))
            {
                HandlePosition(line);
            }
            else if (line.StartsWith("go"))
            {
                HandleGo(line);
            }
            else if (line == "quit")
            {
                break;
            }
            else if (line == "d" || line == "print")
            {
                _board.Print();
                Console.WriteLine(_board.ToFen());
            }
        }
    }

    private void HandlePosition(string line)
    {
        string rest = line.Substring("position".Length).Trim();
        string[] moveSplit = rest.Split("moves", 2);
        string posPart = moveSplit[0].Trim();

        if (posPart.StartsWith("startpos"))
            _board.SetFen(Board.StartFen);
        else if (posPart.StartsWith("fen"))
            _board.SetFen(posPart.Substring(3).Trim());

        if (moveSplit.Length > 1)
        {
            var moveStrs = moveSplit[1].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            foreach (var ms in moveStrs)
            {
                var legal = MoveGen.GenerateLegalMoves(_board);
                Move? found = null;
                foreach (var m in legal)
                {
                    if (m.ToString() == ms) { found = m; break; }
                }
                if (found.HasValue) _board.MakeMove(found.Value);
            }
        }
    }

    private void HandleGo(string line)
    {
        long timeMs = 5000;
        int maxDepth = 64;

        var tokens = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        bool isWhite = _board.SideToMove == Color.White;
        long wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;

        for (int i = 0; i < tokens.Length; i++)
        {
            switch (tokens[i])
            {
                case "wtime": wtime = long.Parse(tokens[++i]); break;
                case "btime": btime = long.Parse(tokens[++i]); break;
                case "winc": winc = long.Parse(tokens[++i]); break;
                case "binc": binc = long.Parse(tokens[++i]); break;
                case "movetime": movetime = long.Parse(tokens[++i]); break;
                case "depth": maxDepth = int.Parse(tokens[++i]); break;
            }
        }

        if (movetime > 0)
        {
            timeMs = movetime;
        }
        else
        {
            long myTime = isWhite ? wtime : btime;
            long myInc = isWhite ? winc : binc;
            if (myTime > 0)
                timeMs = myTime / 30 + myInc / 2;
        }

        var search = new Search(_tt);
        Move best = search.FindBestMove(_board, maxDepth, timeMs);
        Console.WriteLine($"bestmove {(best.IsNull ? "0000" : best.ToString())}");
    }
}

}
