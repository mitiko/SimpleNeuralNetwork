using System;

namespace TreskaAi
{
    public static class ActivationFunctions
    {
        public static double[] Tanh(this double[] vector)
        {
            var result = vector;
            for (int i = 0; i < vector.Length; i++)
                result[i] = Math.Tanh(vector[i]);

            return result;
        }

        public static double[] DTanh(this double[] vector)
        {
            var result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
                result[i] = 1 - Math.Pow(Math.Tanh(vector[i]), 2);

            return result;
        }
    }
}