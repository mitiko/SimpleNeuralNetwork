namespace NeuralNetwork
{
    public class HiddenLayer : Layer
    {
        public HiddenLayer(LayerConfiguration lc)
        {
            Configure(lc, false);
        }

        public override double[] Output()
        {
            return Network.Multiply(this.Neurons, this.Weights);
        }

        public override void Forward()
        {
            var nextLayer = Network.GetLayerByName(this.NextLayerName, this.Network);
            var output = Output();

            for (int j = 0; j < nextLayer.Neurons.Length - 1; j++)
            {
                // We have -1 because last neuron of every layer is bias
                // Set next layer's neurons input to be this layer's output
                nextLayer.Neurons[j].Input = output[j];
            }

            Network.UpdateLayer(nextLayer);
        }
    }
}