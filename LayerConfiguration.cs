namespace NeuralNetwork
{
    public class LayerConfiguration
    {
        public string Name { get; set; }
        public string NextLayerName { get; set; }
        public int NextLayerNeuronCount { get; set; }
        public int NeuronCount { get; set; }
        public string ActivationFunction { get; set; }

        public LayerConfiguration(string Name, string NextLayerName, int NeuronCount, int NextLayerNeuronCount, string ActivationFunction)
        {
            this.Name = Name;
            this.NextLayerName = NextLayerName;
            this.NeuronCount = NeuronCount;
            this.ActivationFunction = ActivationFunction;
            this.NextLayerNeuronCount = NextLayerNeuronCount;
        }
    }
}