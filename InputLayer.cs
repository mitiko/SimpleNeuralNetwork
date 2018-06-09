namespace NeuralNetwork
{
    public class InputLayer : Layer
    {
        public InputLayer(string name, int neuronCount, string activationFunction, Layer nextLayer = null, Layer previousLayer = null) : 
        base(name, neuronCount, activationFunction, nextLayer, previousLayer) { }
        
        public override double[] Output()
        {
            return Network.Multiply(this.Neurons, this.Weights);
        }

        public override void Forward()
        {
            var output = Output();

            for (int j = 0; j < this.NextLayer.Neurons.Length - 1; j++)
                this.NextLayer.Neurons.Input[j] = output[j];
                
            // We have -1 because last neuron of every layer is bias
            // Set next layer's neurons input to be this layer's output
        }
    }
}