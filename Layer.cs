using System;

namespace NeuralNetwork
{
    public class Layer : ILayer
    {
        #region Properties
        public string Name { get; protected set; }
        public LayerConfiguration LayerConfiguration { get; protected set; }
        public string NextLayerName { get; protected set; }
        public Neuron[] Neurons { get; protected set; }
        public double[][] Weights { get; protected set; }
        public Network Network { get; set; }
        #endregion

        #region Configuration
        public void Configure(LayerConfiguration lc, bool isLayerLast)
        {
            this.LayerConfiguration = lc;
            this.Name = lc.Name;
            this.Neurons = new Neuron[lc.NeuronCount + 1]; // +1 for bias(es)
            // Generate Neurons and their activation functions
            for (int z = 0; z < this.Neurons.Length; z++)
            {
                this.Neurons[z] = new Neuron() { ActivationFunction = lc.ActivationFunction };
            }
            // Although the last layer has a bias, it is left unused
            if(!isLayerLast)
            {
                this.Neurons[this.Neurons.Length - 1].Input = 1.0d; // Set bias
                // We are now sure there exists next layer
                this.NextLayerName = lc.NextLayerName; // To calculate the matrix of weights
                this.Weights = new double[this.Neurons.Length][];
                // Generate weights (randomly)
                for (int i = 0; i < this.Neurons.Length; i++)
                {
                    // Bias is not considered when transmitting the next layer neuron count
                    // This is because the next layer's neurons haven't been set yet
                    // That's why we don't -1
                    this.Weights[i] = new double[lc.NextLayerNeuronCount];
                    for (int j = 0; j < Weights[i].Length; j++)
                    {
                        Weights[i][j] = GenerateWeight(new Random());
                    }
                }
            }
        }

        private double GenerateWeight(Random rnd)
        {
            // Note: we pass the rnd object as a parameter, so 
            // the garbage collector doesn't create and destroy it i*j times
            return rnd.NextDouble();
        }
        #endregion

        #region Interface Methods
        public virtual double[] Output()
        {
            return new double[0];
        }

        public virtual void Forward()
        {
        }
        #endregion
    }
}