using System.Collections.Generic;

namespace ChessBot
{
    internal static class PrecomputedMagics
    {
        public static readonly int[] RookShifts = { 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 54, 53, 53, 54, 53, 53, 54, 54, 54, 53, 53, 54, 53, 53, 54, 53, 53, 54, 54, 54, 53, 52, 54, 53, 53, 53, 53, 54, 53, 52, 53, 54, 54, 53, 53, 54, 53, 53, 54, 54, 54, 53, 53, 54, 53, 52, 53, 53, 53, 53, 53, 53, 52 };
        public static readonly int[] BishopShifts = { 58, 60, 59, 59, 59, 59, 60, 58, 60, 59, 59, 59, 59, 59, 59, 60, 59, 59, 57, 57, 57, 57, 59, 59, 59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 57, 57, 57, 59, 59, 60, 60, 59, 59, 59, 59, 60, 60, 58, 60, 59, 59, 59, 59, 59, 58 };

        public static readonly ulong[] RookMagics = { 468374916371625120, 18428729537625841661, 2531023729696186408, 6093370314119450896, 13830552789156493815, 16134110446239088507, 12677615322350354425, 5404321144167858432, 2111097758984580, 18428720740584907710, 17293734603602787839, 4938760079889530922, 7699325603589095390, 9078693890218258431, 578149610753690728, 9496543503900033792, 1155209038552629657, 9224076274589515780, 1835781998207181184, 509120063316431138, 16634043024132535807, 18446673631917146111, 9623686630121410312, 4648737361302392899, 738591182849868645, 1732936432546219272, 2400543327507449856, 5188164365601475096, 10414575345181196316, 1162492212166789136, 9396848738060210946, 622413200109881612, 7998357718131801918, 7719627227008073923, 16181433497662382080, 18441958655457754079, 1267153596645440, 18446726464209379263, 1214021438038606600, 4650128814733526084, 9656144899867951104, 18444421868610287615, 3695311799139303489, 10597006226145476632, 18436046904206950398, 18446726472933277663, 3458977943764860944, 39125045590687766, 9227453435446560384, 6476955465732358656, 1270314852531077632, 2882448553461416064, 11547238928203796481, 1856618300822323264, 2573991788166144, 4936544992551831040, 13690941749405253631, 15852669863439351807, 18302628748190527413, 12682135449552027479, 13830554446930287982, 18302628782487371519, 7924083509981736956, 4734295326018586370 };
        public static readonly ulong[] BishopMagics = { 16509839532542417919, 14391803910955204223, 1848771770702627364, 347925068195328958, 5189277761285652493, 3750937732777063343, 18429848470517967340, 17870072066711748607, 16715520087474960373, 2459353627279607168, 7061705824611107232, 8089129053103260512, 7414579821471224013, 9520647030890121554, 17142940634164625405, 9187037984654475102, 4933695867036173873, 3035992416931960321, 15052160563071165696, 5876081268917084809, 1153484746652717320, 6365855841584713735, 2463646859659644933, 1453259901463176960, 9808859429721908488, 2829141021535244552, 576619101540319252, 5804014844877275314, 4774660099383771136, 328785038479458864, 2360590652863023124, 569550314443282, 17563974527758635567, 11698101887533589556, 5764964460729992192, 6953579832080335136, 1318441160687747328, 8090717009753444376, 16751172641200572929, 5558033503209157252, 17100156536247493656, 7899286223048400564, 4845135427956654145, 2368485888099072, 2399033289953272320, 6976678428284034058, 3134241565013966284, 8661609558376259840, 17275805361393991679, 15391050065516657151, 11529206229534274423, 9876416274250600448, 16432792402597134585, 11975705497012863580, 11457135419348969979, 9763749252098620046, 16960553411078512574, 15563877356819111679, 14994736884583272463, 9441297368950544394, 14537646123432199168, 9888547162215157388, 18140215579194907366, 18374682062228545019 };
    }

    internal static class MagicHelper
    {
        public static ulong[] CreateAllBlockerBitboards(ulong movementMask)
        {
            var indices = new List<int>();
            for (int i = 0; i < 64; i++)
            {
                if (((movementMask >> i) & 1UL) != 0)
                    indices.Add(i);
            }

            int patternCount = 1 << indices.Count;
            var patterns = new ulong[patternCount];
            for (int patternIndex = 0; patternIndex < patternCount; patternIndex++)
            {
                ulong bitboard = 0;
                for (int bitIndex = 0; bitIndex < indices.Count; bitIndex++)
                {
                    if (((patternIndex >> bitIndex) & 1) != 0)
                        bitboard |= 1UL << indices[bitIndex];
                }
                patterns[patternIndex] = bitboard;
            }
            return patterns;
        }

        public static ulong CreateMovementMask(int squareIndex, bool orthogonal)
        {
            ulong mask = 0;
            int file = Sq.File(squareIndex);
            int rank = Sq.Rank(squareIndex);
            int[][] directions = orthogonal
                ? new[] { new[] { 0, 1 }, new[] { 0, -1 }, new[] { 1, 0 }, new[] { -1, 0 } }
                : new[] { new[] { 1, 1 }, new[] { -1, 1 }, new[] { 1, -1 }, new[] { -1, -1 } };

            foreach (var dir in directions)
            {
                for (int dist = 1; dist < 8; dist++)
                {
                    int nf = file + dir[0] * dist;
                    int nr = rank + dir[1] * dist;
                    if (nf < 0 || nf >= 8 || nr < 0 || nr >= 8)
                        break;
                    mask |= Bits.Bit(Sq.Make(nf, nr));
                }
            }

            return mask;
        }

        public static ulong LegalMoveBitboardFromBlockers(int startSquare, ulong blockerBitboard, bool orthogonal)
        {
            ulong moves = 0;
            int file = Sq.File(startSquare);
            int rank = Sq.Rank(startSquare);
            int[][] directions = orthogonal
                ? new[] { new[] { 0, 1 }, new[] { 0, -1 }, new[] { 1, 0 }, new[] { -1, 0 } }
                : new[] { new[] { 1, 1 }, new[] { -1, 1 }, new[] { 1, -1 }, new[] { -1, -1 } };

            foreach (var dir in directions)
            {
                for (int dist = 1; dist < 8; dist++)
                {
                    int nf = file + dir[0] * dist;
                    int nr = rank + dir[1] * dist;
                    if (nf < 0 || nf >= 8 || nr < 0 || nr >= 8)
                        break;

                    int sq = Sq.Make(nf, nr);
                    moves |= Bits.Bit(sq);
                    if (((blockerBitboard >> sq) & 1UL) != 0)
                        break;
                }
            }

            return moves;
        }
    }

    internal static class Magic
    {
        public static readonly ulong[] RookMask = new ulong[64];
        public static readonly ulong[] BishopMask = new ulong[64];
        public static readonly ulong[][] RookAttacks = new ulong[64][];
        public static readonly ulong[][] BishopAttacks = new ulong[64][];

        static Magic()
        {
            for (int square = 0; square < 64; square++)
            {
                RookMask[square] = MagicHelper.CreateMovementMask(square, true);
                BishopMask[square] = MagicHelper.CreateMovementMask(square, false);
            }

            for (int square = 0; square < 64; square++)
            {
                RookAttacks[square] = CreateTable(square, true, PrecomputedMagics.RookMagics[square], PrecomputedMagics.RookShifts[square]);
                BishopAttacks[square] = CreateTable(square, false, PrecomputedMagics.BishopMagics[square], PrecomputedMagics.BishopShifts[square]);
            }
        }

        public static ulong GetRookAttacks(int square, ulong blockers)
        {
            ulong key = ((blockers & RookMask[square]) * PrecomputedMagics.RookMagics[square]) >> PrecomputedMagics.RookShifts[square];
            if (key >= (ulong)RookAttacks[square].Length)
                return ComputeRookAttacksFallback(square, blockers);
            return RookAttacks[square][(int)key];
        }

        public static ulong GetBishopAttacks(int square, ulong blockers)
        {
            ulong key = ((blockers & BishopMask[square]) * PrecomputedMagics.BishopMagics[square]) >> PrecomputedMagics.BishopShifts[square];
            if (key >= (ulong)BishopAttacks[square].Length)
                return ComputeBishopAttacksFallback(square, blockers);
            return BishopAttacks[square][(int)key];
        }

        public static ulong GetSliderAttacks(int square, ulong blockers, bool orthogonal)
            => orthogonal ? GetRookAttacks(square, blockers) : GetBishopAttacks(square, blockers);

        private static ulong[] CreateTable(int square, bool orthogonal, ulong magic, int shift)
        {
            int lookupSize = 1 << (64 - shift);
            var table = new ulong[lookupSize];
            var blockerPatterns = MagicHelper.CreateAllBlockerBitboards(orthogonal ? RookMask[square] : BishopMask[square]);

            foreach (var pattern in blockerPatterns)
            {
                int index = (int)((pattern * magic) >> shift);
                if (index < 0 || index >= table.Length)
                    continue;
                table[index] = MagicHelper.LegalMoveBitboardFromBlockers(square, pattern, orthogonal);
            }

            return table;
        }

        private static ulong ComputeRookAttacksFallback(int square, ulong blockers)
        {
            ulong attacks = 0;
            int file = Sq.File(square);
            int rank = Sq.Rank(square);

            for (int r = rank + 1; r < 8; r++)
            {
                int sq = Sq.Make(file, r);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            for (int r = rank - 1; r >= 0; r--)
            {
                int sq = Sq.Make(file, r);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            for (int f = file + 1; f < 8; f++)
            {
                int sq = Sq.Make(f, rank);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            for (int f = file - 1; f >= 0; f--)
            {
                int sq = Sq.Make(f, rank);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            return attacks;
        }

        private static ulong ComputeBishopAttacksFallback(int square, ulong blockers)
        {
            ulong attacks = 0;
            int file = Sq.File(square);
            int rank = Sq.Rank(square);

            for (int f = file + 1, r = rank + 1; f < 8 && r < 8; f++, r++)
            {
                int sq = Sq.Make(f, r);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            for (int f = file - 1, r = rank + 1; f >= 0 && r < 8; f--, r++)
            {
                int sq = Sq.Make(f, r);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            for (int f = file + 1, r = rank - 1; f < 8 && r >= 0; f++, r--)
            {
                int sq = Sq.Make(f, r);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            for (int f = file - 1, r = rank - 1; f >= 0 && r >= 0; f--, r--)
            {
                int sq = Sq.Make(f, r);
                attacks |= Bits.Bit(sq);
                if (((blockers >> sq) & 1UL) != 0) break;
            }
            return attacks;
        }
    }
}
