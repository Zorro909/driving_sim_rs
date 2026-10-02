using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace AILearnsToDrive.Lib.Evolutions;

// The parts of the game's NeuralNetwork that Forward uses, plus the current Forward for comparison.
public partial record NeuralNetwork
{
    public int[] Shape { get; }
    public Matrix<double>[] Weights { get; }
    public Vector<double>[] Biases { get; }

    public NeuralNetwork(int[] shape, Matrix<double>[] weights, Vector<double>[] biases)
    {
        Shape = shape;
        Weights = weights;
        Biases = biases;
    }

    // The current implementation, unchanged.
    public double[] ForwardMathNet(double[] input)
    {
        Matrix<double> matrix = DenseVector.OfArray(input).ToRowMatrix();
        for (int i = 0; i < Shape.Length - 1; i++)
        {
            matrix = matrix.Multiply(Weights[i]).Add(Biases[i].ToRowMatrix()).PointwiseTanh();
        }
        return matrix.Row(0).ToArray();
    }
}
