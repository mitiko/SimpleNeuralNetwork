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
            for (int i = 0; i < result.Length; i++)
            {
                this.Neurons.Activate();
                result[i] = this.Neurons.Output[i];
                // TODO: Try directly assigning arrays -> unstable build
            }
            return result;
        }
    }
}