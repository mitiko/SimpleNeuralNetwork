namespace NeuralNetwork
{
    public class OutputLayer : Layer
    {
        public OutputLayer(string name, int neuronCount, string activationFunction, Layer nextLayer = null, Layer previousLayer = null)
        {
            Configure(name, neuronCount, activationFunction, nextLayer, previousLayer, false);
        }

        public override double[] Output()
        {
            // Skip the unused bias
            var result = new double[this.Neurons.Length - 1];
            for (int i = 0; i < result.Length; i++)
            {
                this.Neurons[i].Activate();
                result[i] = this.Neurons[i].Output;
            }
            return result;
        }
    }
}