using System;

namespace NeuralNetwork
{
    public class Neuron
    {
        #region Properties
        public double Input { get; internal set; }
        public double Output { get; private set; }
        public double Error { get; set; }
        public string ActivationFunction { get; set; }
        #endregion

        #region Methods
        public void Activate()
        {
            switch (this.ActivationFunction)
            {
                case "linear":
                    this.Output = this.Input;
                    break;
                case "logistic":
                case "sigmoid":
                    this.Output = Sigmoid(this.Input);
                    break;
                case "tanh":
                    this.Output = HyperbolicTangent(this.Input);
                    break;
                case "logisticprime":
                case "sigmoidprime":
                    this.Output = SigmoidPrime(this.Input);
                    break;
                case "easteregg":
                case "bob":
                    this.Output = Bob(this.Input);
                    break;
                default:
                    Console.WriteLine("Activation function  not supported, using 2 * sigmoid(1.45x) - 1");
                    this.Output = Bob(this.Input);
                    break;
            }
        }

        public double DeActivate()
        {
            // Get the derivateive of the activation function
            // use this.Output as input
            double result = 1.0d;
            switch (this.ActivationFunction)
            {
                case "linear":
                    break;
                case "logistic":
                case "sigmoid":
                    result = DeSigmoid(this.Output);
                    break;
                case "tanh":
                    result = DeHyperbolicTangent(this.Output);
                    break;
                case "logisticprime":
                case "sigmoidprime":
                    result = DeSigmoidPrime(this.Output);
                    break;
                case "easteregg":
                case "bob":
                    result = DeBob(this.Output);
                    break;
                default:
                    Console.WriteLine("Activation function  not supported, using 2 * sigmoid(1.45x) - 1");
                    result = DeBob(this.Output);
                    break;
            }

            return result;
        }
        #endregion

        #region Activation Functions
        public static double Sigmoid(double value)
        {
            double s = Math.Exp(value);
            return s / (1.0d + s);
        }

        public static double DeSigmoid(double value)
        {
            double s = Sigmoid(value);
            return s * (1 - s);
        }

        public static double SigmoidPrime(double value)
        {
            return Sigmoid(value) * 2 -1;
        }

        public static double DeSigmoidPrime(double value)
        {
            return 2 * DeSigmoid(value);
        }

        public static double HyperbolicTangent(double value)
        {
            return 2 * Sigmoid(2 * value) - 1;
        }

        public static double DeHyperbolicTangent(double value)
        {
            return 4 * DeSigmoid(2 * value);
        }

        public static double Bob(double value)
        {
            return SigmoidPrime(1.45 * value);
        }

        public static double DeBob(double value)
        {
            return 2.9 * DeSigmoid(1.45 * value);
        }
        #endregion
    }
}