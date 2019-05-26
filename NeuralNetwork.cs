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
        public DataHelper _dataHelper { get; set; }

        public NeuralNetwork(DataHelper dataHelper = null)
        {
            this.Layers = new List<Layer>();
            this.IsBuilt = false;
            if(dataHelper == null)
                this._dataHelper = new DataHelper();
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

        public void SetLearningRate(double learningRate = 0.1)
        {
            this.Layers.ForEach(l => l.LearningRate = learningRate);
        }

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

            // Add bias
            this.InputLayer.Input = input.Append(1).ToArray();
            var layer = this.InputLayer;
            while (layer != this.OutputLayer)
            {
                // Add bias
                layer.NextLayer.Input = layer.Forward().Append(1).ToArray();
                layer = layer.NextLayer;
            }

            return layer.Input.Take(layer.NeuronCount).ToArray();
        }

        public void Backpropagate(double[] expectedResult, Func<double[], double[], double[]> lossFunction)
        {
            if (!this.IsBuilt)
            {
                Console.WriteLine("[CRITICAL] Build network before backpropagating");
                throw new InvalidOperationException();
            }

            this.OutputLayer.Error = lossFunction.Invoke(this.OutputLayer.Input.Take(this.OutputLayer.NeuronCount).ToArray(), expectedResult);
            var layer = this.OutputLayer;
            do
            {
                layer = layer.PreviousLayer;
                layer.Backward();
            }
            while (layer != this.InputLayer);
        }

        public void Train(
            long epochs, Func<double[], double[], double[]> lossFunction,
            IEnumerable<(double[], double[])> reader,
            string sampleCount = "all", int spinSpeed = 40)
        {
            var s = Stopwatch.StartNew();
            for (long i = 0; i < epochs; i++)
            {
                int samplesProcessed = 0;
                double summedLoss = 0;
                Console.WriteLine($"Epoch {i + 1}/{epochs}:");
                foreach (var (input, output) in reader)
                {
                    var result = this.FeedForward(input);
                    this.Backpropagate(output, lossFunction);
                    var loss = this.OutputLayer.Error.Select(y => Math.Abs(y)).Sum() / this.OutputLayer.NeuronCount;
                    summedLoss += loss;
                    samplesProcessed++;

                    this._dataHelper.LogTrainingInformation(samplesProcessed, s, spinSpeed, sampleCount);
                }
                this._dataHelper.LogEpochInformation(summedLoss, samplesProcessed);
            }
        }

        public void Test(IEnumerable<(double[], double[])> reader)
        {
            Console.WriteLine("Test:");
            int samplesProcessed = 0;
            double summedLoss = 0;
            foreach (var (input, output) in reader)
            {
                var result = this.FeedForward(input);
                summedLoss += this.OutputLayer.Input.SimpleLoss(output).Select(y => Math.Abs(y)).Sum() / this.OutputLayer.NeuronCount;
                samplesProcessed++;
            }
            this._dataHelper.LogTestResults(summedLoss, samplesProcessed);
        }

        public void Save(string fileName)
        {
            using (var sw = new StreamWriter(fileName))
            {
                sw.WriteLine("v2");
                sw.WriteLine(this.Layers.Count);
                foreach (var layer in this.Layers)
                {
                    sw.WriteLine(layer.Id);
                    sw.WriteLine(layer.NeuronCount);
                    // sw.WriteLine(layer.LearningRate);
                    // sw.WriteLine(layer.ActivationFunction);
                    if (layer.NextLayer != null)
                    {
                        sw.WriteLine(layer.NextLayer.NeuronCount);
                        foreach (var w in layer.Weights)
                        {
                            foreach (var weight in w)
                            {
                                sw.WriteLine($"{weight}");
                            }
                        }
                    }
                }
            }
        }

        public void Load(string fileName)
        {
            using (var sr = new StreamReader(fileName))
            {
                if (sr.ReadLine() != "v2") throw new Exception();
                var layers = int.Parse(sr.ReadLine());
                for (int i = 0; i < layers; i++)
                {
                    var id = sr.ReadLine();
                    var neurons = int.Parse(sr.ReadLine());
                    this.AddLayer(new Layer(neurons, id));
                    if (i < layers - 1)
                    {
                        var nextLayerNeurons = int.Parse(sr.ReadLine());
                        this.Layers.Last().Weights = new double[neurons + 1][];

                        for (int a = 0; a < neurons + 1; a++)
                        {
                            this.Layers.Last().Weights[a] = new double[nextLayerNeurons];
                            for (int b = 0; b < nextLayerNeurons; b++)
                            {
                                this.Layers.Last().Weights[a][b] = double.Parse(sr.ReadLine());
                            }
                        }
                    }
                }
            }
            var s = "";
            for (int m = 0; m < this.Layers.Count - 1; m++)
            {
                s += this.Layers[m].Id + ";";
            }
            s+=this.Layers.Last().Id;
            this.SetInput(this.Layers.First().Id);
            this.SetOutput(this.Layers.Last().Id);
            for (int i = 0; i < this.Layers.Count - 1; i++)
                this.Layers[i].NextLayer = this.Layers[i+1];

            for (int i = 1; i < this.Layers.Count; i++)
                this.Layers[i].PreviousLayer = this.Layers[i-1];

            this.IsBuilt = true;
        }
    }
}