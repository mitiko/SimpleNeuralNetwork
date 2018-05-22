namespace NeuralNetwork
{
    public interface ILayer
    {
        Network Network { get; set; }

        double[] Output();
        void Forward();
    }
}