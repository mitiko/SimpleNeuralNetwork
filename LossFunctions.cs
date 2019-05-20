using System;
using System.Linq;

namespace TreskaAi
{
    public static class LossFunctions
    {
        #region Regression losses

        public static double MeanAbsoluteError(this double[] guess, double[] expected)
        {
            double result = 0;
            for (int i = 0; i < expected.Length; i++)
                result += Math.Abs(expected[i] - guess[i]);

            return result / expected.Length;
        }

        public static double RootMeanSquaredError(this double[] guess, double[] expected)
        {
            double result = 0;
            for (int i = 0; i < expected.Length; i++)
                result += Math.Pow(expected[i] - guess[i], 2);

            return Math.Sqrt(result) / expected.Length;
        }

        public static double MeanSquaredError(this double[] guess, double[] expected)
        {
            double result = 0;
            for (int i = 0; i < expected.Length; i++)
                result += Math.Pow(expected[i] - guess[i], 2);

            return result / (2 * expected.Length);
        }

        public static double HuberLoss(this double[] guess, double[] expected, double delta = 5)
        {
            double result = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                var diff = expected[i] - guess[i];
                if (Math.Abs(diff) <= delta)
                    result += 0.5 * Math.Pow(diff, 2);
                else
                    result += delta * diff - 0.5 * Math.Pow(delta, 2);
            }

            return result / expected.Length;
        }

        public static double LogCoshLoss(this double[] guess, double[] expected)
        {
            double result = 0;
            for (int i = 0; i < expected.Length; i++)
                result += Math.Log(Math.Cosh(expected[i] - guess[i]));

            return result / expected.Length;
        }

        #endregion

        #region Classification losses

        public static double[] CrossEntropyLoss(this double[] guess, double[] expected)
        {
            // TODO: Find why this is wrong
            for (int i = 0; i < expected.Length; i++)
                expected[i] = 0 - expected[i] * Math.Log(1e-15 + guess[i]);

            return expected;
        }

        public static double[] SimpleLoss(this double[] guess, double[] expected)
        {
            for (int i = 0; i < expected.Length; i++)
                expected[i] = expected[i] - guess[i];

            return expected;
        }

        #endregion
    }
}