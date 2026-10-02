// Replacement for NeuralNetwork.Forward (AILearnsToDrive.Lib.Evolutions/NeuralNetwork.cs).
// Paste the two fields and the method into the NeuralNetwork record, replacing the current Forward.
// The file is written as a partial record only so the test project can compile it unchanged.

using System;
using MathNet.Numerics.LinearAlgebra.Double;

namespace AILearnsToDrive.Lib.Evolutions;

public partial record NeuralNetwork
{
    // Scratch buffers for the hidden layers. Forward runs on several worker threads at once,
    // so they are per thread.
    [ThreadStatic] private static double[]? _forwardBufferA;
    [ThreadStatic] private static double[]? _forwardBufferB;

    /// <summary>
    /// Same result as the MathNet version, bit for bit, without its temporaries (it allocated
    /// four matrices per layer and call). It repeats the arithmetic of MathNet's managed provider
    /// in the same order: for each output k, num = 0.0; num += x[r] * W[r, k] over the inputs;
    /// the product matrix holds 0.0 + num; then the bias is added and tanh applied.
    /// </summary>
    public double[] Forward(double[] input)
    {
        if (input.Length != Shape[0])
        {
            throw new ArgumentException($"Expected {Shape[0]} inputs, got {input.Length}.", nameof(input));
        }
        int maxWidth = 0;
        for (int i = 1; i < Shape.Length; i++)
        {
            maxWidth = Math.Max(maxWidth, Shape[i]);
        }
        double[] next = _forwardBufferA is { } bufferA && bufferA.Length >= maxWidth ? bufferA : (_forwardBufferA = new double[maxWidth]);
        double[] spare = _forwardBufferB is { } bufferB && bufferB.Length >= maxWidth ? bufferB : (_forwardBufferB = new double[maxWidth]);

        ReadOnlySpan<double> current = input;
        for (int i = 0; i < Shape.Length - 1; i++)
        {
            // Weights[i] is inputs x outputs and stored column-major, so the weights into output k
            // are contiguous: W[r, k] = weights[k * rows + r].
            double[] weights = Weights[i].AsColumnMajorArray() ?? Weights[i].ToColumnMajorArray();
            double[] biases = Biases[i] is DenseVector dense ? dense.Values : Biases[i].ToArray();
            int rows = Weights[i].RowCount, cols = Weights[i].ColumnCount;
            for (int k = 0; k < cols; k++)
            {
                ReadOnlySpan<double> column = weights.AsSpan(k * rows, rows);
                double num = 0.0;
                for (int r = 0; r < rows; r++)
                {
                    num += current[r] * column[r];
                }
                // 0.0 + num as in MathNet: it turns -0.0 into +0.0.
                next[k] = Math.Tanh((0.0 + num) + biases[k]);
            }
            current = next.AsSpan(0, cols);
            (next, spare) = (spare, next);
        }
        return current.ToArray();
    }
}
