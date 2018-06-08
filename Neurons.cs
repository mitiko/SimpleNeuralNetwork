using System;

namespace NeuralNetwork
{
    public class Neurons
    {
        #region Properties
        public double[] Input { get; protected internal set; }
        public double[] Output { get; private set; }
        public double[] Error { get; private set; }
        public string ActivationFunction { get; set; }
        public int Length { get; private set; }
        #endregion

        #region Methods
        public Neurons(int numberOfNeurons, string activationFunction)
        {
            this.ActivationFunction = activationFunction;
            this.Length = numberOfNeurons;
            this.Input = new double[this.Length];
            this.Output = new double[this.Length];
            this.Error = new double[this.Length];

            for (int i = 0; i < this.Length; i++)
            {
                this.Input[i] = 0;
                this.Output[i] = 0;
                this.Error[i] = 0;
            }

            // Bias input is always 1
            this.Input[this.Length - 1] = 1;
        }

        public Func<double, double> Activation()
        {
            Func<double, double> result = null;
            switch (this.ActivationFunction)
            {
                case "linear":
                    result = new Func<double, double>(x => x);
                    break;
                case "logistic":
                case "sigmoid":
                    result = Sigmoid;
                    break;
                case "tanh":
                    result = HyperbolicTangent;
                    break;
                case "logisticprime":
                case "sigmoidprime":
                    result = SigmoidPrime;
                    break;
                case "easteregg":
                case "bob":
                    result = Bob;
                    break;
                default:
                    Console.WriteLine("Activation function not supported, using default: sigmoid");
                    result = Sigmoid;
                    break;
            }

            return result;
        }

        public Func<double, double> DeActivatation()
        {
            // Get the derivateive of the activation function
            // use this.Output as input
            Func<double, double> result = null;
            switch (this.ActivationFunction)
            {
                case "linear":
                    break;
                case "logistic":
                case "sigmoid":
                    result = DeSigmoid;
                    break;
                case "tanh":
                    result = DeHyperbolicTangent;
                    break;
                case "logisticprime":
                case "sigmoidprime":
                    result = DeSigmoidPrime;
                    break;
                case "easteregg":
                case "bob":
                    result = DeBob;
                    break;
                default:
                    Console.WriteLine("Activation function  not supported, using default: sigmoid");
                    result = DeSigmoid;
                    break;
            }

            return result;
        }

        public void Activate()
        {
            var af = Activation();

            for (int i = 0; i < this.Length; i++)
            {
                this.Output[i] = af.Invoke(this.Input[i]);
            }
        }

        public double[] DeActivate()
        {
            var df = DeActivatation();
            double[] result = new double[this.Length];

            for (int i = 0; i < this.Length; i++)
            {
                result[i] = df.Invoke(this.Input[i]);
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
            double s = Math.Exp(value) / (1.0d + Math.Exp(value));
            return s * (1 - s);
        }

        public static double SigmoidPrime(double value)
        {
            double s = Math.Exp(value);
            return s / (1.0d + s) * 2 - 1;
        }

        public static double DeSigmoidPrime(double value)
        {
            double s = Math.Exp(value) / (1.0d + Math.Exp(value));
            return s * (1 - s) * 2;
        }

        public static double HyperbolicTangent(double value)
        {
            double s = Math.Exp(2 * value);
            return s / (1.0d + s) * 2 - 1;
        }

        public static double DeHyperbolicTangent(double value)
        {
            double s = Math.Exp(2 * value) / (1.0d + Math.Exp(2 * value));
            return 4 * s * (1 - s);
        }

        public static double Bob(double value)
        {
            double s = Math.Exp(1.45 * value);
            return s / (1.0d + s) * 2 - 1;
        }

        public static double DeBob(double value)
        {
            double s = Math.Exp(1.45 * value) / (1.0d + Math.Exp(1.45 * value));
            return 2.9 * s * (1 - s);
        }
        #endregion
    }
}