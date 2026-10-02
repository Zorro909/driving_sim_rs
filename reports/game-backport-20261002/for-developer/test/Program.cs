using System.Diagnostics;
using AILearnsToDrive.Lib.Evolutions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Providers.LinearAlgebra;

Console.WriteLine($"MathNet provider: {LinearAlgebraControl.Provider}");

// 1. Exactness: random networks and inputs, including signed zeros, huge, tiny and subnormal values.
var rng = new Random(12345);
double[] special = { 0.0, -0.0, 1.0, -1.0, 1e-300, -1e-300, 1e300, -1e300, double.Epsilon, 0.5 };
double Value(double scale) => rng.Next(8) == 0 ? special[rng.Next(special.Length)] : (rng.NextDouble() * 2 - 1) * scale;
long outputs = 0, mismatches = 0;
for (int trial = 0; trial < 20000; trial++)
{
    int layers = rng.Next(1, 11);
    int[] shape = new int[layers + 1];
    for (int i = 0; i <= layers; i++) shape[i] = rng.Next(1, i == 0 ? 40 : 70);
    double scale = rng.Next(3) switch { 0 => 0.1, 1 => 2.0, _ => 50.0 };
    var network = RandomNetwork(shape, () => Value(scale));
    for (int k = 0; k < 5; k++)
    {
        double[] input = new double[shape[0]];
        for (int i = 0; i < input.Length; i++) input[i] = Value(scale);
        double[] expected = network.ForwardMathNet(input), actual = network.Forward(input);
        outputs += expected.Length;
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.DoubleToInt64Bits(expected[i]) != BitConverter.DoubleToInt64Bits(actual[i]) && mismatches++ < 5)
            {
                Console.WriteLine($"  mismatch, shape [{string.Join(",", shape)}], output {i}: {actual[i]:R} vs {expected[i]:R}");
            }
        }
    }
}
Console.WriteLine($"Exactness: {outputs:N0} outputs compared, {mismatches} differ.");

// 2. Speed: the shape of a typical trained network (20 inputs, 8 hidden layers, 5 outputs).
int[] rally = { 20, 16, 16, 16, 16, 12, 12, 12, 8, 5 };
var net = RandomNetwork(rally, () => rng.NextDouble() * 2 - 1);
double[] x = Enumerable.Range(0, 20).Select(_ => rng.NextDouble()).ToArray();
double Time(Func<double[], double[]> forward)
{
    const int calls = 200_000;
    for (int i = 0; i < 20_000; i++) forward(x);  // warm up
    var watch = Stopwatch.StartNew();
    for (int i = 0; i < calls; i++) forward(x);
    return watch.Elapsed.TotalMilliseconds * 1000 / calls;
}
long before = GC.GetAllocatedBytesForCurrentThread();
net.ForwardMathNet(x);
long mathNetBytes = GC.GetAllocatedBytesForCurrentThread() - before;
before = GC.GetAllocatedBytesForCurrentThread();
net.Forward(x);
long newBytes = GC.GetAllocatedBytesForCurrentThread() - before;
double oldUs = Time(net.ForwardMathNet), newUs = Time(net.Forward);
Console.WriteLine($"Speed, shape [{string.Join(",", rally)}], one thread:");
Console.WriteLine($"  MathNet: {oldUs:F2} µs per call, {mathNetBytes:N0} bytes allocated");
Console.WriteLine($"  new:     {newUs:F2} µs per call, {newBytes:N0} bytes allocated ({oldUs / newUs:F1}x faster)");
return mismatches == 0 ? 0 : 1;

static NeuralNetwork RandomNetwork(int[] shape, Func<double> value)
{
    int layers = shape.Length - 1;
    var weights = new Matrix<double>[layers];
    var biases = new Vector<double>[layers];
    for (int l = 0; l < layers; l++)
    {
        weights[l] = DenseMatrix.Create(shape[l], shape[l + 1], (_, _) => value());
        biases[l] = DenseVector.Create(shape[l + 1], _ => value());
    }
    return new NeuralNetwork(shape, weights, biases);
}
