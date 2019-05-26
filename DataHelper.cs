using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SimpleNeuralNetwork
{
    public class DataHelper
    {
        public static IEnumerable<(double[] x, double[] y)> ReadCsvLines(string fileName, IEnumerable<int> inputIndexes, IEnumerable<int> outputIndexes)
        {
            bool firstLine = true;
            string line;

            using (StreamReader sr = new StreamReader(fileName, System.Text.Encoding.Default))
            {
                while ((line = sr.ReadLine()) != null)
                {
                    if (firstLine)
                    {
                        line = sr.ReadLine(); // Skip first line
                        firstLine = false;
                    }

                    var sample = line.Split(',');
                    var input = inputIndexes.Select(x => double.Parse(sample[x])).ToArray();
                    var output = outputIndexes.Select(x => double.Parse(sample[x])).ToArray();

                    yield return (input, output);
                }
            }
        }

        public virtual void LogTrainingInformation(int samplesProcessed, Stopwatch s, int spinSpeed, string sampleCount)
        {
            string loading = "/";
            if (samplesProcessed % spinSpeed == 0) loading = "/";
            else if (samplesProcessed % spinSpeed == (spinSpeed / 4) * 1) loading = "-";
            else if (samplesProcessed % spinSpeed == (spinSpeed / 4) * 2) loading = "\\";
            else if (samplesProcessed % spinSpeed == (spinSpeed / 4) * 3) loading = "|";
            Console.Write($"\r{loading} Time: {s.Elapsed.ToString(@"dd\.hh\:mm\:ss")} Samples: {samplesProcessed}/{sampleCount}");
        }

        public virtual void LogEpochInformation(double summedLoss, int samplesProcessed)
        {
            var averageLoss = summedLoss / samplesProcessed;
            Console.WriteLine();
            Console.WriteLine($"Loss: {averageLoss}");
            Console.WriteLine($"Accuracy: {1 - averageLoss}");
            Console.WriteLine("--------------------------------------");
        }

        public virtual void LogTestResults(double summedLoss, int samplesProcessed)
        {
            var averageLoss = summedLoss / samplesProcessed;
            Console.WriteLine($"Loss: {averageLoss}");
            Console.WriteLine($"Accuracy: {1 - averageLoss}");
            Console.WriteLine("--------------------------------------");
        }
    }
}