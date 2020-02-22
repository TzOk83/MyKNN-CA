using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;

namespace kNN
{
    public class ValuePair<T1, T2>
    {
        public T1 Item1 { get; set; }
        public T2 Item2 { get; set; }
        public ValuePair() { }
        public ValuePair(T1 Item1, T2 Item2)
        {
            this.Item1 = Item1;
            this.Item2 = Item2;
        }
    }

    class Program
    {
        public static string FPKNN(ref List<Tuple<double?[], string>> training, Tuple<double?[], string> test, int k, out ValuePair<string, double>[] suppval, bool minorityBoost, bool useRangeCheck, bool distWeight)
        {
            //Deklaracje zmiennych lokalnych
            //...najbliższa i najdalsza odległość
            double d0, dk;
            //...licznik rozpatrywanych instancji
            int neiCount;
            //...licznik cech wektora testowego
            int validFeatures = 0;
            //...licznik wartości cechy
            int validValues;
            //...lista klas
            string[] classLbls = training.Select(x => x.Item2).Distinct().OrderBy(x => x).ToArray();
            //...wektor liczności klas
            int[] classCnt = new int[classLbls.Length];
            for (int i = 0; i < classLbls.Length; i++)
            {
                classCnt[i] = training.Count(x => x.Item2 == classLbls[i]);
            }
            //...tablica funkcji wsparcia
            suppval = new ValuePair<string, double>[classLbls.GetLength(0)];
            for (int i = 0; i < classLbls.GetLength(0); i++)
                suppval[i] = new ValuePair<string, double>(classLbls[i], 0d);
            //...tablice wartości środkowych i dolnego/górnego progu
            double[] medVal = new double[classLbls.GetLength(0)];
            double[] lowLimit = new double[classLbls.GetLength(0)];
            double[] uppLimit = new double[classLbls.GetLength(0)];
            bool inLimits;
            int medIdx, idxRange;
            //...lista wartości atrybutu
            List<double> attValues = new List<double>();
            //...tablica odległości
            List<ValuePair<string, double>> distVector = new List<ValuePair<string, double>>();
            //Pętla po cechach
            for (int feature = 0; feature < test.Item1.GetLength(0); feature++)
            {
                validValues = 0;
                //Jeśli atrybut w klasyfikowanym wektorze posiada wartość...
                if (test.Item1[feature].HasValue)
                {
                    inLimits = true;
                    if (useRangeCheck)
                    {
                        //Wyznaczenie zakresów zamienności atrybutu dla każdej z klas
                        inLimits = false;
                        for (int i = 0; i < classLbls.GetLength(0); i++)
                        {
                            attValues.Clear();
                            for (int row = 0; row < training.Count; row++)
                                if (training[row].Item2 == classLbls[i] && training[row].Item1[feature].HasValue)
                                    attValues.Add(training[row].Item1[feature].Value);
                            attValues.Sort();
                            medIdx = attValues.Count / 2;
                            idxRange = (int)(medIdx * 0.75); //Zakres zmienności - między 1 a 3 kwartylem
                            medVal[i] = attValues[medIdx]; //Wartość środkowa
                            lowLimit[i] = attValues[medIdx - idxRange]; //Wartość minimalna
                            uppLimit[i] = attValues[medIdx + idxRange]; //Wartość maksymalna
                            if (test.Item1[feature].Value >= lowLimit[i] && test.Item1[feature].Value <= uppLimit[i])
                                inLimits = true; //Wartość atrybutu mieści się w zakresie zmieności tego atrybutu w zbiorze referencyjnym
                        }
                    }
                    //Jeśli wartość atrybutu klasyfikowanego wektora mieści się w zakresie zmienności atrybutu
                    //dla poszczególnych klas w zbiorze referencyjnym...
                    if (inLimits)
                    {
                        //Zliczanie niepustych cech obiektu klasyfikowanego
                        validFeatures++;
                        //Wyznaczenie odległości od wektora testowego
                        distVector.Clear();
                        for (int row = 0; row < training.Count; row++)
                            if (training[row].Item1[feature].HasValue)
                            {
                                validValues++;
                                distVector.Add(new ValuePair<string, double>(training[row].Item2, Math.Abs((double)training[row].Item1[feature] - (double)test.Item1[feature])));
                            }
                        //Sortowanie wyniku w/g odległości od wektora testowego
                        distVector = distVector.OrderBy(x => x.Item2).ToList();
                        //Rozstrzyganie remisów na odległościach - rozszerznie "okna analizy"
                        for (neiCount = 0; neiCount < k; )
                        {
                            //Wyznaczenie ile wśród k pierwszych najmniejszych odległości jest sobie równych
                            neiCount += distVector.Count(x => x.Item2.Equals(distVector.ElementAt(neiCount).Item2));
                        }
                        //Wyznaczenie odległości najbliższego i najdlaszego z pośród k najbliższych sąsiadw
                        d0 = distVector[0].Item2;
                        //dk = distVector[neiCount - 1].Item2;
                        dk = distVector[distVector.Count - 1].Item2;
                        //w = (dMax - d)/(dMax - dMin)
                        //Wyznaczenie wartości wsparcia dla klas z opcjonalnym premiowaniem klas mniejszościowych
                        for (int i = 0; i < classLbls.GetLength(0); i++)
                            suppval[i].Item2 += (minorityBoost ? (1 - (double)classCnt[i] / classCnt.Sum()) : 1d) * (distVector.Take(neiCount).Where(x => x.Item1 == classLbls[i]).Sum(x => distWeight ? (dk - x.Item2) / (dk - d0 + 0.001) : 1d) / (double)neiCount);
                            //suppval[i].Item2 += (validValues / training.Count) * (minorityBoost ? (1 - (double)classCnt[i] / classCnt.Sum()) : 1d) * (distVector.Take(neiCount).Count(x => x.Item1 == classLbls[i]) / (double)neiCount);
                            //suppval[i].Item2 += (minorityBoost ? (1 - (double)classCnt[i] / classCnt.Sum()) : 1d) * (distVector.Take(neiCount).Count(x => x.Item1 == classLbls[i]) / (double)neiCount);
                    }
                }
            }
            //Normalizacja wartości wsparcia
            for (int i = 0; i < classLbls.GetLength(0); i++)
                suppval[i].Item2 = suppval[i].Item2 / validFeatures / (minorityBoost ? (1 - (double)classCnt[i] / classCnt.Sum()) : 1d);
            //Wyłonienie zwycięzcy
            return suppval.OrderByDescending(x => x.Item2).First().Item1;
        }

        public static string vectorToString(Tuple<double?[], string> vector)
        {
            string result = "";
            foreach (double? value in vector.Item1)
            {
                result += value.ToString() + ",";
            }
            result += vector.Item2;
            return result;
        }

        public static string supvalToString(ValuePair<string, double>[] supval)
        {
            string result = "";
            foreach (ValuePair<string, double> row in supval)
            {
                result += (Double.IsNaN(row.Item2) ? "" : row.Item2.ToString("0.000")) + ",";
            }
            return result;
        }

        public static void parseCSV(string fname, out List<Tuple<double?[], string>> data, out string[] header)
        {
            data = new List<Tuple<double?[], string>>();
            const char delim = ',';
            string line;
            string[] parts;
            int fieldCnt;
            double?[] attribs;
            string classLbl;
            using (StreamReader reader = new StreamReader(fname))
            {
                // Odczytanie nagłówka - określenie liczby kolumn
                line = reader.ReadLine();
                parts = line.Split(delim);
                fieldCnt = parts.Length;
                // Wyodrębnienie nagłówka
                header = new string[fieldCnt];
                parts.CopyTo(header, 0);
                // Parsowanie pliku do listy
                while (true)
                {
                    line = reader.ReadLine();
                    if (line == null)
                    {
                        break;
                    }
                    parts = line.Split(delim);
                    attribs = new double?[fieldCnt - 1];
                    for (int i = 0; i < fieldCnt - 1; i++)
                    {
                        if (parts[i].Equals(""))
                            attribs[i] = null;
                        else
                            attribs[i] = double.Parse(parts[i]);
                    }
                    classLbl = parts[fieldCnt - 1].Replace("\"", string.Empty);
                    data.Add(new Tuple<double?[], string>(attribs, classLbl));
                }
            }
        }

        public static double N_FPKNN(ref List<Tuple<double?[], string>> trainset, ref List<Tuple<double?[], string>> testset, ref string[] header, string out_fname, int k, bool minBoost, bool useRangeCheck, bool distWeight, out int TPC, out int FPC)
        {
            //Klasyfikacja tablicy danych wejściowych
            ValuePair<string, double>[] supval;
            Tuple<double?[], string> tvec;
            TPC = 0;
            FPC = 0;
            string winner;
            string line;

            using (StreamWriter writer = new StreamWriter(out_fname))
            {
                // Zapisanie nagłówka do pliku wynikowego
                writer.Write(string.Join(",", header));
                for (int row = 0; row < testset.Count; row++)
                {
                    //Wyjęcie wektora testowego
                    tvec = testset[row];
                    //Wywołanie klasyfikatora
                    winner = FPKNN(ref trainset, tvec, k, out supval, minBoost, useRangeCheck, distWeight);
                    // Uzupełnienie nagłówka przy pierwszym przejściu pętli
                    if (row == 0)
                    {
                        foreach (ValuePair<string, double> hrow in supval)
                            writer.Write("," + hrow.Item1);
                        writer.WriteLine(",winner");
                    }
                    //Wyjście klasyfikatora
                    line = vectorToString(tvec) + "," + supvalToString(supval) + winner;
                    writer.WriteLine(line);
                    //Statystyki
                    if (winner == tvec.Item2)
                        TPC++;
                    else
                        FPC++;
                }
            }
            return TPC * 100d / (TPC + FPC);
        }

        public static double CV_FPKNN(ref List<Tuple<double?[], string>> trainset, ref string[] header, string fname, int k, bool minBoost, bool useProbabl, bool distWeight, out int TPC, out int FPC)
        {
            //Kroswalidacja leave-one-out
            ValuePair<string, double>[] supval;
            Tuple<double?[], string> tvec;
            TPC = 0;
            FPC = 0;
            string winner;
            string line;
            using (StreamWriter writer = new StreamWriter(fname))
            {
                // Zapisanie nagłówna do pliku wynikowego
                writer.Write(string.Join(",", header));
                for (int row = 0; row < trainset.Count; row++)
                {
                    //Wyjęcie wektora testowego
                    tvec = trainset[row];
                    //Usunięcie go ze zbioru trenującego
                    trainset.RemoveAt(row);
                    //Wywołanie klasyfikatora
                    winner = FPKNN(ref trainset, tvec, k, out supval, minBoost, useProbabl, distWeight);
                    // Uzupełnienie nagłówka przy pierwszym przejściu pętli
                    if (row == 0)
                    {
                        foreach (ValuePair<string, double> hrow in supval)
                            writer.Write("," + hrow.Item1);
                        writer.WriteLine(",winner");
                    }
                    //Przywrócenie wektora testującego do zbioru trenującego
                    trainset.Insert(row, tvec);
                    //Wyjście klasyfikatora
                    line = vectorToString(tvec) + "," + supvalToString(supval) + winner;
                    writer.WriteLine(line);
                    //Statystyki
                    if (winner == tvec.Item2)
                        TPC++;
                    else
                        FPC++;
                }
            }
            return TPC * 100d / (TPC + FPC);
        }

        public static double CV2_FPKNN(ref List<Tuple<double?[], string>> trainset, ref List<Tuple<double?[], string>> testset, ref string[] header, string fname, int k, bool minBoost, bool useProbabl, bool distWeight, out int TPC, out int FPC)
        {
            //Kroswalidacja leave-one-out
            ValuePair<string, double>[] supval;
            Tuple<double?[], string> tvec, trvec;
            TPC = 0;
            FPC = 0;
            string winner;
            string line;
            using (StreamWriter writer = new StreamWriter(fname))
            {
                // Zapisanie nagłówna do pliku wynikowego
                writer.Write(string.Join(",", header));
                for (int row = 0; row < trainset.Count; row++)
                {
                    //Wyjęcie wektora testowego
                    tvec = testset[row];
                    //...i odpowaiadającego mu wektora ze zbioru trenującego
                    trvec = trainset[row];
                    //Usunięcie wyjętego wektora ze zbioru trenującego
                    trainset.RemoveAt(row);
                    //Wywołanie klasyfikatora
                    winner = FPKNN(ref trainset, tvec, k, out supval, minBoost, useProbabl, distWeight);
                    // Uzupełnienie nagłówka przy pierwszym przejściu pętli
                    if (row == 0)
                    {
                        foreach (ValuePair<string, double> hrow in supval)
                            writer.Write("," + hrow.Item1);
                        writer.WriteLine(",winner");
                    }
                    //Przywrócenie wektora testującego do zbioru trenującego
                    trainset.Insert(row, trvec);
                    //Wyjście klasyfikatora
                    line = vectorToString(tvec) + "," + supvalToString(supval) + winner;
                    writer.WriteLine(line);
                    //Statystyki
                    if (winner == tvec.Item2)
                        TPC++;
                    else
                        FPC++;
                }
            }
            return TPC * 100d / (TPC + FPC);
        }

        // knn.exe <k> <MinBoost>
        // k - minimal nearest neighbors window size (1 - 99+)
        // MinBoost - use minority class promoting algorithm (t/f)
        // RangeCheck - use only attributes within the range of reference set +/-75% (t/f)
        // CrossValidate - do a trainset crossvalidation using leave one out method (t/2/f)
        // DistanceWeighting - weight neighbor importance by its distance (t/f)
        public static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US", false);

            int TPC;
            int FPC;

            List<Tuple<double?[], string>> trainset;
            List<Tuple<double?[], string>> testset;
            string[] header;

            const string trainfile = @"train.csv";
            const string testfile = @"test.csv";
            const string ofile = @"out.csv";

            int k = int.Parse(args[0]);
            bool useMinBoost = (args[1] == "t" ? true : false);
            bool useRangeCheck = (args[2] == "t" ? true : false);
            bool doCrossVal = (args[3] == "t" ? true : false);
            bool do2SetCrossVal = (args[3] == "2" ? true : false);
            bool distWeight = (args[4] == "t" ? true : false);

            parseCSV(trainfile, out trainset, out header);
            if (doCrossVal)
            {
                CV_FPKNN(ref trainset, ref header, ofile, k, useMinBoost, useRangeCheck, distWeight, out TPC, out FPC);
            }
            else if (do2SetCrossVal)
            {
                parseCSV(testfile, out testset, out header);
                CV2_FPKNN(ref trainset, ref testset, ref header, ofile, k, useMinBoost, useRangeCheck, distWeight, out TPC, out FPC);
            }
            else
            {
                parseCSV(testfile, out testset, out header);
                N_FPKNN(ref trainset, ref testset, ref header, ofile, k, useMinBoost, useRangeCheck, distWeight, out TPC, out FPC);
            }
        }
    }
}