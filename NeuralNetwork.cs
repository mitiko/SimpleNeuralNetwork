using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace TreskaAi
{
    public sealed class NeuralNetwork
    {
        private List<Layer> Layers { get; set; }
        public Layer InputLayer { get; private set; }
        public Layer OutputLayer { get; private set; }
        public bool IsBuilt { get; private set; }
        public string Flow { get; private set; }

        public NeuralNetwork()
        {
            this.Layers = new List<Layer>();
            this.IsBuilt = false;
        }

        public void AddLayer(Layer layer)
        {
            if (this.IsBuilt == false)
            {
                this.Layers.Add(layer);
            }
            else
            {
                Console.WriteLine("[WARNING] Layer cannot be added after network is built");
            }
        }

        public void SetLearningRate(double learningRate = 0.1) =>
            this.Layers.ForEach(l => l.LearningRate = learningRate);

        public void SetInput(string inputLayer)
        {
            try
            {
                var layer = this.Layers.SingleOrDefault(l => l.Id == inputLayer);

                if (layer == null)
                    throw new InvalidOperationException();
                else
                    this.InputLayer = layer;
            }
            catch (InvalidOperationException)
            {
                // TODO: Allow multiple input layers for multiple data sources
                Console.WriteLine($"[CRITICAL] Multiple or no input layers with id {inputLayer} exist");
            }
        }

        public void SetOutput(string outputLayer)
        {
            try
            {
                var layer = this.Layers.SingleOrDefault(l => l.Id == outputLayer);

                if (layer == null)
                    throw new InvalidOperationException();
                else
                    this.OutputLayer = layer;
            }
            catch (InvalidOperationException)
            {
                // TODO: Allow multiple output layers for multiple data sources
                Console.WriteLine($"[CRITICAL] Multiple or no input layers with id {outputLayer} exist");
            }
        }

        public void LinkLayers(string flow)
        {
            this.Flow = flow;

            try
            {
                var layerStack = this.Flow
                    .Split(';')
                    .Select(layerId => this.Layers
                        .SingleOrDefault(l => l.Id == layerId))
                    .ToList();

                foreach (var layer in layerStack)
                {
                    int index = layerStack.IndexOf(layer);
                    if (index != layerStack.Count - 1)
                        layer.NextLayer = layerStack[index + 1];
                    if (index != 0)
                        layer.PreviousLayer = layerStack[index - 1];

                    layer.Setup();
                }
            }
            catch (InvalidOperationException)
            {
                Console.WriteLine("[WARNING] Currently RNNs aren't supported");
            }
        }

        public void Build()
        {
            if (this.IsBuilt == false)
            {
                if (this.InputLayer == null || this.OutputLayer == null)
                {
                    Console.WriteLine("[CRITICAL] Input or Output layer not set");
                    return;
                }
                if (this.InputLayer.NextLayer == null)
                {
                    Console.WriteLine("[CRITICAL] Input layer not connected or layers aren't linked");
                    return;
                }

                this.IsBuilt = true;
            }
            else
            {
                Console.WriteLine("[WARNING] Cannot build a built network");
            }
        }

        public double[] FeedForward(double[] input)
        {
            if (!this.IsBuilt)
            {
                Console.WriteLine("[CRITICAL] Build network before feedforwarding");
                throw new InvalidOperationException();
            }

            this.InputLayer.Input = input.Append(1).ToArray();
            var layer = this.InputLayer;
            while (layer != this.OutputLayer)
            {
                layer.NextLayer.Input = layer.Forward().Append(1).ToArray();
                layer = layer.NextLayer;
            }

            return layer.Input;
        }

        public void Backpropagate(double[] expectedResult)
        {
            this.OutputLayer.Error = this.OutputLayer.Input.CrossEntropyLoss(expectedResult);
            var layer = this.OutputLayer.PreviousLayer;
            do
            {
                layer.Backward();
                layer = layer.PreviousLayer;
            }
            while (layer != this.InputLayer);
        }

        public void Train(double[][] inputs, double[][] outputs, long epochs)
        {
            Console.WriteLine("Training data:");
            var s = Stopwatch.StartNew();
            for (long i = 0; i < epochs; i++)
            {
                Console.WriteLine($"Epoch {i + 1}:");
                double maxLoss = -100;
                double minLoss =  100;
                double averageLoss = 0;

                for (int j = 0; j < inputs.Length; j++)
                {
                    var result = this.FeedForward(inputs[j]);
                    this.Backpropagate(outputs[j]);
                    var loss = this.OutputLayer.Error.Select(y => Math.Abs(y)).Sum() / this.OutputLayer.NeuronCount;
                    if (maxLoss < loss) maxLoss = loss;
                    if (minLoss > loss) minLoss = loss;
                    averageLoss = averageLoss * ((double)j / (j + 1)) + loss / (j + 1);
                }

                Console.WriteLine($"Time elapsed: {s.ElapsedMilliseconds}");
                Console.WriteLine($"Loss: {averageLoss}");
                Console.WriteLine($"MinLoss: {minLoss}");
                Console.WriteLine($"MaxLoss: {maxLoss}");
                Console.WriteLine("--------------------------------------");
            }

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        public void Train(string fileName, IEnumerable<int> inputIndexes, IEnumerable<int> outputIndexes, long epochs, string sampleCount = "all", int spinSpeed = 40)
        {
            var s = Stopwatch.StartNew();
            for (long i = 0; i < epochs; i++)
            {
                bool firstLine = true;
                string line;
                Console.WriteLine($"Epoch {i + 1}/{epochs}:");
                double maxLoss = -100;
                double minLoss =  100;
                double averageLoss = 0;
                int j = 0;

                using (StreamReader sr = new StreamReader(fileName, System.Text.Encoding.Default))
                {
                    string loading = "";
                    while((line = sr.ReadLine()) != null)
                    {
                        #region Extract data
                        if(firstLine)
                        {
                            line = sr.ReadLine();
                            firstLine = false;
                        }

                        var sample = line.Split(',');
                        var input = inputIndexes.Select(x => double.Parse(sample[x])).ToArray();
                        var output = outputIndexes.Select(x => double.Parse(sample[x])).ToArray();
                        #endregion

                        var result = this.FeedForward(input);
                        this.Backpropagate(output);
                        var loss = this.OutputLayer.Error.Select(y => Math.Abs(y)).Sum() / this.OutputLayer.NeuronCount;
                        if (maxLoss < loss) maxLoss = loss;
                        if (minLoss > loss) minLoss = loss;
                        averageLoss = averageLoss * ((double)j / (j + 1)) + loss / (j + 1);
                        j++;

                        if(j % spinSpeed == 0) loading = "/";
                        else if(j % spinSpeed == (spinSpeed / 4) * 1) loading = "-";
                        else if(j % spinSpeed == (spinSpeed / 4) * 2) loading = "\\";
                        else if(j % spinSpeed == (spinSpeed / 4) * 3) loading = "|";
                        Console.Write($"\r{loading} Time: {s.Elapsed.ToString(@"dd\.hh\:mm\:ss")} Samples: {j}/{sampleCount}");
                    }
                }

                Console.WriteLine();
                Console.WriteLine($"Loss: {averageLoss}");
                Console.WriteLine($"Accuracy: {1 - averageLoss}");
                Console.WriteLine($"MinLoss: {minLoss}");
                Console.WriteLine($"MaxLoss: {maxLoss}");
                Console.WriteLine("--------------------------------------");
            }
        }

        public void Test(double[][] inputs, double[][] outputs)
        {
            double averageLoss = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var result = this.FeedForward(inputs[i]);
                var loss = this.OutputLayer.Input.CrossEntropyLoss(outputs[i]).Select(y => Math.Abs(y)).Sum() / this.OutputLayer.NeuronCount;
                averageLoss = averageLoss * ((double)i / (i + 1)) + loss / (i + 1);
            }

            Console.WriteLine("Test data:");
            Console.WriteLine($"Loss: {averageLoss}");
            Console.WriteLine("--------------------------------------");
        }

        public void Test(string fileName, IEnumerable<int> inputIndexes, IEnumerable<int> outputIndexes)
        {
            bool firstLine = true;
            string line;
            double averageLoss = 0;
            int j = 0;

            using (StreamReader sr = new StreamReader(fileName, System.Text.Encoding.Default))
            {
                while((line = sr.ReadLine()) != null)
                {
                    #region Extract data
                    if(firstLine)
                    {
                        line = sr.ReadLine();
                        firstLine = false;
                    }

                    var sample = line.Split(',');
                    var input = inputIndexes.Select(x => double.Parse(sample[x])).ToArray();
                    var output = outputIndexes.Select(x => double.Parse(sample[x])).ToArray();
                    #endregion

                    var result = this.FeedForward(input);
                    var loss = this.OutputLayer.Input.CrossEntropyLoss(output).Select(y => Math.Abs(y)).Sum() / this.OutputLayer.NeuronCount;
                    averageLoss = averageLoss * ((double)j / (j + 1)) + loss / (j + 1);
                    j++;
                }
            }

            Console.WriteLine($"Loss: {averageLoss}");
            Console.WriteLine("--------------------------------------");
        }
    }
}