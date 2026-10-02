using ChessBot;
// dotnet build ChessBot.csproj -c Release
// cd "c:\Users\colin\OneDrive\Documents\Unity projects and Assets\UnityChess\NewScripts"
// dotnet run --project ChessBot.csproj -c Release -- cli
if (args.Length > 0 && args[0] == "cli")
{
    RunCli();
}
else
{
    new Uci().Run();
}

static void RunCli()
{
    var board = new Board();
    var tt = new TranspositionTable(1024);

    Console.WriteLine("ChessBot CLI mode. Enter moves like 'e2e4', 'go' to let the bot move, 'quit' to exit.");
    while (true)
    {
        board.Print();
        Console.WriteLine(board.ToFen());
        Console.Write("> ");
        string? input = Console.ReadLine();
        if (input == null || input == "quit") break;
        input = input.Trim();

        if (input == "go")
        {
            var search = new Search(tt);
            Move best = search.FindBestMove(board, 40, 6000);
            if (best.IsNull)
            {
                Console.WriteLine("No legal moves - game over.");
                break;
            }
            Console.WriteLine($"Bot plays: {best}");
            board.MakeMove(best);
            continue;
        }

        if (input.Equals("game", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine(board.ToPgn());
            continue;
        }

        if (input.StartsWith("fen ", StringComparison.OrdinalIgnoreCase))
        {
            string fen = input[4..].Trim();
            if (fen.Length == 0)
            {
                Console.WriteLine("Usage: fen <position-fen>");
            }
            else
            {
                try
                {
                    board.SetFen(fen);
                    Console.WriteLine("Loaded position from FEN.");
                }
                catch
                {
                    Console.WriteLine("Invalid FEN. Use a standard chess FEN string.");
                }
            }
            continue;
        }

        var legal = MoveGen.GenerateLegalMoves(board);
        Move? found = null;
        foreach (var m in legal)
        {
            if (m.ToString() == input) { found = m; break; }
        }
        if (found.HasValue)
        {
            board.MakeMove(found.Value);
        }
        else
        {
            Console.WriteLine("Illegal or unrecognized move. Legal moves: " +
                string.Join(" ", legal.Select(m => m.ToString())));
        }
    }
}
