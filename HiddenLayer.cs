namespace NeuralNetwork
{
    public class HiddenLayer : Layer
    {
        public HiddenLayer(string name, int neuronCount, string activationFunction, Layer nextLayer = null, Layer previousLayer = null) : 
        base(name, neuronCount, activationFunction, nextLayer, previousLayer) { }

        public override double[] Output()
        {
            this.Neurons.Activate();
            return Network.Multiply(this.Neurons.Output, this.Weights);
        }

        public override void Forward()
        {
            var output = Output();

            for (int j = 0; j < this.NextLayer.Neurons.Length - 1; j++)
            {
                // We have -1 because last neuron of every layer is bias
                // Set next layer's neurons input to be this layer's output
                this.NextLayer.Neurons.Input[j] = output[j];
                // TODO: Try directly assigning arrays -> unstable build
            }
        }
    }
}