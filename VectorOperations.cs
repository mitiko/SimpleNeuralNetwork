using System;
using System.Linq;

namespace SimpleNeuralNetwork
{
    public class VectorOperations
    {
        public double[] Value { get; set; }

        public static explicit operator VectorOperations (double[] x) => new VectorOperations() { Value = x };

        public static double[] operator + (VectorOperations x, double[] y)
        {
            if (x.Value.Length != y.Length)
            {
                Console.WriteLine("[CRITICAL] Vectors dimensions don't match.");
                Console.WriteLine($"[INFO] Vector x: 1x{x.Value.Length}  Vector y: 1x{y.Length}");

                throw new InvalidOperationException("Addition inapplicable");
            }

            for (int i = 0; i < y.Length; i++)
            {
                y[i] += x.Value[i];
            }

            return y;
        }

        public static double[] operator - (VectorOperations x, double[] y)
        {
            if (x.Value.Length != y.Length)
            {
                Console.WriteLine("[CRITICAL] Vectors dimensions don't match.");
                Console.WriteLine($"[INFO] Vector x: 1x{x.Value.Length}  Vector y: 1x{y.Length}");

                throw new InvalidOperationException("Substraction inapplicable");
            }

            for (int i = 0; i < y.Length; i++)
            {
                x.Value[i] -= y[i];
            }

            return x.Value;
        }

        public static double[] operator * (double a, VectorOperations x)
        {
            for (int i = 0; i < x.Value.Length; i++)
                x.Value[i] *= a;

            return x.Value;
        }
    }

    public class MatrixOperations
    {
        public double[][] Value { get; set; }

        public static explicit operator MatrixOperations (double[][] x) => new MatrixOperations() { Value = x };

        public static double[][] operator + (MatrixOperations x, double[][] y)
        {
            if (x.Value.Length != y.Length || x.Value[0].Length != y[0].Length)
            {
                Console.WriteLine("[CRITICAL] Matrices dimensions don't match.");
                Console.WriteLine($"[INFO] Matrix x: {x.Value.Length}x{x.Value[0].Length}  matrix y: {y.Length}x{y[0].Length}");

                throw new InvalidOperationException("Addition inapplicable");
            }

            for (int i = 0; i < y.Length; i++)
                for (int j = 0; j < y[0].Length; j++)
                    y[i][j] += x.Value[i][j];

            return y;
        }

        public static double[][] operator - (MatrixOperations x, double[][] y)
        {
            if (x.Value.Length != y.Length || x.Value[0].Length != y[0].Length)
            {
                Console.WriteLine("[CRITICAL] Matrices dimensions don't match.");
                Console.WriteLine($"[INFO] Matrix x: {x.Value.Length}x{x.Value[0].Length}  Matrix y: {y.Length}x{y[0].Length}");

                throw new InvalidOperationException("Substraction inapplicable");
            }

            for (int i = 0; i < y.Length; i++)
                for (int j = 0; j < y[0].Length; j++)
                    x.Value[i][j] -= y[i][j];

            return x.Value;
        }

        public static double[][] operator * (double a, MatrixOperations x)
        {
            for (int i = 0; i < x.Value.Length; i++)
                for (int j = 0; j < x.Value[0].Length; j++)
                    x.Value[i][j] *= a;

            return x.Value;
        }
    }

    public static class VectorMultiplication
    {
        public static double[] Multiply(this double[] vector, double[][] matrix)
        {
            if (vector.Length != matrix.Length)
            {
                Console.WriteLine("[CRITICAL] Matrix and vector dimensions don't match.");
                Console.WriteLine("[INFO] Have you linked the layers properly?");
                Console.WriteLine($"[INFO] Vector: 1x{vector.Length}  Matrix: {matrix.Length}x{matrix[0].Length}");

                throw new InvalidOperationException("Multiplication inapplicable");
            }

            var result = new double[matrix[0].Length];

            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[0].Length; j++)
                    result[j] += vector[i] * matrix[i][j];

            return result;
        }

        public static double[][] Multiply(this double[] x, double[] y)
        {
            var result = new double[x.Length][];
            for (int m = 0; m < result.Length; m++)
                result[m] = new double[y.Length];

            for (int i = 0; i < x.Length; i++)
                for (int j = 0; j < y.Length; j++)
                    result[i][j] = x[i] * y[j];

            return result;
        }

        public static double[][] Multiply(this double[][] x, double[][] y)
        {
            if (x[0].Length != y.Length)
            {
                Console.WriteLine("[CRITICAL] Metrices dimensions don't match.");
                Console.WriteLine($"[INFO] Matrix x: {x.Length}x{x[0].Length}  Matrix y: {y.Length}x{y[0].Length}");

                throw new InvalidOperationException("Multiplication inapplicable");
            }

            var result = new double[x.Length][];
            for (int m = 0; m < result.Length; m++)
                result[m] = new double[y[0].Length];

            for (int i = 0; i < x.Length; i++)
                for (int j = 0; j < y[0].Length; j++)
                    for (int k = 0; k < y.Length; k++)
                        result[i][j] += x[i][k] * y[k][j];

            return result;
        }

        public static double[] TransposeMultiply(this double[] vector, double[][] matrix)
        {
            if (vector.Length != matrix[0].Length)
            {
                Console.WriteLine("[CRITICAL] Matrix and vector dimensions don't match.");
                Console.WriteLine("[INFO] Have you linked the layers properly?");
                Console.WriteLine($"[INFO] Vector: 1x{vector.Length}  Matrix: {matrix.Length}x{matrix[0].Length}");

                throw new InvalidOperationException("Multiplication inapplicable");
            }

            var result = new double[matrix.Length];

            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[0].Length; j++)
                    result[i] += vector[j] * matrix[i][j];

            return result;
        }

        public static double[][] Transpose(this double[][] matrix)
        {
            var result = new double[matrix[0].Length][];
            for (int m = 0; m < result.Length; m++)
                result[m] = new double[matrix.Length];

            for (int i = 0; i < matrix.Length; i++)
                for (int j = 0; j < matrix[0].Length; j++)
                    result[j][i] = matrix[i][j];

            return result;
        }

        public static double[] HadamardMultiply(this double[] x, double[] y)
        {
            if (x.Length != y.Length)
            {
                Console.WriteLine("[CRITICAL] Vector 1 and vector 2 dimensions don't match.");
                Console.WriteLine($"[INFO] Vector 1: 1x{x.Length}  Vector 2: 1x{y.Length}");

                throw new InvalidOperationException("Hadamard multiplication inapplicable");
            }

            for (int i = 0; i < x.Length; i++)
                x[i] *= y[i];

            return x;
        }

        public static double[][] HadamardMultiply(this double[][] x, double[][] y)
        {
            if (x.Length != y.Length || x.Length == 0 || y.Length == 0)
            {
                Console.WriteLine("[CRITICAL] Vector 1 and vector 2 dimensions don't match.");
                Console.WriteLine($"[INFO] Vector 1: 1x{x.Length}  Vector 2: 1x{y.Length}");

                throw new InvalidOperationException("Hadamard multiplication inapplicable");
            }
            else if (x[0].Length != y[0].Length || x[0].Length == 0 || y[0].Length == 0)
            {
                Console.WriteLine("[CRITICAL] Vector 1 and vector 2 dimensions don't match.");
                Console.WriteLine($"[INFO] Vector 1: 1x{x.Length}  Vector 2: 1x{y.Length}");

                throw new InvalidOperationException("Hadamard multiplication inapplicable");
            }

            for (int j = 0; j < x[0].Length; j++)
                for (int i = 0; i < x.Length; i++)
                    x[i][j] *= y[i][j];

            return x;
        }
    }
}