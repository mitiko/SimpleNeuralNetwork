using System;
using System.Linq;

namespace TreskaAi
{
    public class Layer
    {
        public int NeuronCount { get; private set; }
        public string Id { get; private set; }
        public double LearningRate { get; set; }

        public Layer NextLayer { get; internal protected set; }
        public Layer PreviousLayer { get; internal protected set; }

        public double[] Input { get; internal protected set; }
        public double[] Output { get; internal protected set; }
        public double[] Error { get; internal protected set; }
        public double[][] Weights { get; internal protected set; }

        public Layer(int neurons, string id)
        {
            // TODO: Add activation function
            this.NeuronCount = neurons;
            this.Input = new double[neurons + 1];
            this.Error = new double[neurons];
            this.Id = id;
        }

        public virtual void Setup()
        {
            if (this.NextLayer != null)
            {
                this.Output = new double[this.NextLayer.NeuronCount];
                this.Weights = new double[this.NeuronCount + 1][];
                var rnd = new Random();
                for (int i = 0; i < this.Weights.Length; i++)
                {
                    this.Weights[i] = new double[this.NextLayer.NeuronCount];
                    for (int j = 0; j < this.Weights[i].Length; j++)
                    {
                        this.Weights[i][j] = rnd.NextDouble() * 2 - 1;
                    }
                }
            }
        }

        public virtual double[] Forward()
        {
            // Multitply input vector with matrix
            this.Output = this.Input.Multiply(this.Weights);
            // Apply activation function
            return this.Output.Tanh();
        }

        public virtual void Backward()
        {
            // Activation Function Derivative
            var afd = this.Output.DTanh();

            // Calculate error at previous layer
            this.Error = new double[this.Error.Length];

            for (int i = 0; i < this.Error.Length; i++)
                for (int j = 0; j < this.NextLayer.NeuronCount; j++)
                    this.Error[i] += this.Weights[i][j] * afd[j] * this.NextLayer.Error[j];

            // Calculate change to weights
            for (int i = 0; i < this.Input.Length; i++)
                for (int j = 0; j < this.NextLayer.NeuronCount; j++)
                    this.Weights[i][j] += this.LearningRate * this.Input[i] * afd[j] * this.NextLayer.Error[j];
        }
    }
}