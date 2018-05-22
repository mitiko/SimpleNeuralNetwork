namespace NeuralNetwork
{
    public class OutputLayer : Layer
    {
        public OutputLayer(LayerConfiguration lc)
        {
            Configure(lc, true);
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