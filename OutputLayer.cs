namespace NeuralNetwork
{
    public class OutputLayer : Layer
    {
        public OutputLayer(string name, int neuronCount, string activationFunction, Layer nextLayer = null, Layer previousLayer = null) : 
        base(name, neuronCount, activationFunction, nextLayer, previousLayer) { }

        public override double[] Output()
        {
            // Skip the unused bias
            var result = new double[this.Neurons.Length - 1];
            this.Neurons.Activate();

            for (int i = 0; i < result.Length; i++)
                result[i] = this.Neurons.Output[i];
                
            return result;
        }
    }
}