using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace TreskaAi
{
    public static class DataHelpers
    {
        public static IEnumerable<(double[], double[])> ReadCsvLines(string fileName, IEnumerable<int> inputIndexes, IEnumerable<int> outputIndexes)
        {
            bool firstLine = true;
            string line;

            using (StreamReader sr = new StreamReader(fileName, System.Text.Encoding.Default))
            {
                while ((line = sr.ReadLine()) != null)
                {
                    if (firstLine)
                    {
                        line = sr.ReadLine();
                        firstLine = false;
                    }

                    var sample = line.Split(',');
                    var input = inputIndexes.Select(x => double.Parse(sample[x])).ToArray();
                    var output = outputIndexes.Select(x => double.Parse(sample[x])).ToArray();

                    yield return (input, output);
                }
            }
        }

        public static void LogTrainingInformation(double summedLoss, double averageLoss, int samplesProcessed, Stopwatch s, int spinSpeed, string sampleCount)
        {
            string loading = "";
            averageLoss = averageLoss * ((double)samplesProcessed / (samplesProcessed + 1)) + summedLoss / (samplesProcessed + 1);

            if (samplesProcessed % spinSpeed == 0) loading = "/";
            else if (samplesProcessed % spinSpeed == (spinSpeed / 4) * 1) loading = "-";
            else if (samplesProcessed % spinSpeed == (spinSpeed / 4) * 2) loading = "\\";
            else if (samplesProcessed % spinSpeed == (spinSpeed / 4) * 3) loading = "|";
            Console.Write($"\r{loading} Time: {s.Elapsed.ToString(@"dd\.hh\:mm\:ss")} Samples: {samplesProcessed}/{sampleCount}");

            Console.WriteLine();
            Console.WriteLine($"Loss: {averageLoss}");
            Console.WriteLine($"Accuracy: {1 - averageLoss}");
            Console.WriteLine("--------------------------------------");
        }

        public static void LogTestingInformation(double summedLoss, double averageLoss, int samplesProcessed)
        {
            averageLoss = averageLoss * ((double)samplesProcessed / (samplesProcessed + 1)) + summedLoss / (samplesProcessed + 1);
            Console.WriteLine($"Loss: {averageLoss}");
            Console.WriteLine($"Accuracy: {1 - averageLoss}");
            Console.WriteLine("--------------------------------------");
        }
    }
}