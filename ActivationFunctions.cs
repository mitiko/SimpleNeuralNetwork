using System;
using System.Linq;

namespace SimpleNeuralNetwork
{
    public static class ActivationFunctions
    {
        public static double[] Tanh(this double[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = Math.Tanh(vector[i]);

            return vector;
        }

        public static double[][] DTanh(this double[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = 1 - Math.Pow(Math.Tanh(vector[i]), 2);

            return new double[1][] { vector };
        }

        public static double[] ELU(this double[] vector, double alpha = 1)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = vector[i] <= 0 ? alpha * (Math.Exp(vector[i]) - 1) : vector[i];

            return vector;
        }

        public static double[][] DELU(this double[] vector, double alpha = 1)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = vector[i] < 0 ? alpha * (Math.Exp(vector[i])) : 1;

            return new double[1][] { vector };
        }

        public static double[] LReLU(this double[] vector, double alpha  = 0)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = vector[i] < 0 ? alpha * vector[i] : vector[i];

            return vector;
        }

        public static double[][] DLReLU(this double[] vector, double alpha = 0)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = vector[i] < 0 ? alpha : 1;

            return new double[1][] { vector };
        }

        public static double[] Sigmoid(this double[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = 1 / (1 + Math.Exp(vector[i]));

            return vector;
        }

        public static double[][] DSigmoid(this double[] vector)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = Math.Exp(vector[i]) / Math.Pow(1 + Math.Exp(vector[i]), 2);

            return new double[1][] { vector };
        }

        public static double[] Softmax(this double[] vector)
        {
            var max = vector.Max();
            var result = vector.Select(y => Math.Exp(y - max));
            var sum = result.Sum();
            return result.Select(y => y / sum).ToArray();
        }

        public static double[][] DSoftmax(this double[] probabilities)
        {
            var jacobian = new double[probabilities.Length][];
            for (int m = 0; m < jacobian.Length; m++)
                jacobian[m] = new double[jacobian.Length];

            for (int i = 0; i < jacobian.Length; i++)
                for (int j = 0; j < jacobian.Length; j++)
                    jacobian[i][j] = i == j ? probabilities[i] * (1 - probabilities[j]) : - probabilities[i] * probabilities[j];

            return jacobian;
        }
    }
}