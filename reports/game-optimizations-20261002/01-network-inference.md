# 01: Network inference without allocations

Finding 17 in the [overview](README.md). Complexity: medium. Estimated impact: high.

## What the game does now

`NeuralNetwork.Forward` (✓ decompiled):

```csharp
Matrix<double> matrix = DenseVector.OfArray(input).ToRowMatrix();
for (int i = 0; i < Shape.Length - 1; i++)
    matrix = matrix.Multiply(Weights[i]).Add(Biases[i].ToRowMatrix()).PointwiseTanh();
return matrix.Row(0).ToArray();
```

Each layer allocates:

- the product matrix from `Multiply`,
- a row matrix from `Biases[i].ToRowMatrix()`,
- the sum matrix from `Add`,
- the result matrix from `PointwiseTanh`.

The input vector, its row matrix, the final row vector and the output array add four more allocations per call. The default network has 8 layers, so each inference allocates more than 30 objects. Each car runs inference up to 60 times per second, depending on the batch setting. With hundreds of cars at high time scales, that is millions of short-lived objects per second.

Other costs per call:

- MathNet checks dimensions and dispatches to a linear algebra provider. ? For a 1×16 by 16×16 product (256 multiply-adds), that fixed overhead is likely comparable to the arithmetic itself.
- `SensorCollection.SetInput` fills a `NeuralNetworkInput` wrapper. `NeuralNetworkOutput` then wraps the result, and every read goes through a `Dictionary` (finding 1).
- ? Gen0 garbage collections pause every thread, including the `Parallel.For` workers that run inference. Their cost was not measured here, but it grows with the allocation rate above.

## What the Rust port does

- `network.rs` stores all parameters of a network in one `Vec<f64>`. Each layer is laid out as the weight matrix (row = input, column = output), then the biases. ✓ This is the same order as the game's `NeuralNetwork.GetVector`, so a network can be converted either way by copying the array.
- `forward_into_scalar` uses two scratch buffers that are reused across calls and across cars on the same thread:

  ```rust
  next.fill(0.0);
  for (i, &value) in current.iter().enumerate() {
      let row = &weights[i * outputs..(i + 1) * outputs];
      for j in 0..outputs { next[j] += value * row[j]; }
  }
  for j in 0..outputs { next[j] = game_tanh(next[j] + bias[j]); }
  ```

  ✓ The result is bit-identical to the game. Each output is a sum over inputs in input order, starting at 0.0, and the bias is added afterwards. That is the order MathNet's managed provider uses for a row vector times a matrix. The fidelity experiment (`../experiments/driving_sim_rs_fidelity/`) matches every recorded network output exactly on three game episodes.
- `network_simd.rs` computes 4 outputs at once in a 256-bit register (`_mm256_mul_pd`, then `_mm256_add_pd`). Each lane runs the same operations in the same order as the scalar code, so the result is identical. The kernel does not use FMA, because a fused multiply-add rounds once instead of twice and would change the result.

## Suggested C# implementation

### 1. Flat parameters

Keep `Weights` and `Biases` for saving, the UI and crossover if needed. Add one flat array built from `GetVector()` whenever the network changes:

```csharp
public sealed class FlatNetwork
{
    public readonly int[] Shape;
    public readonly double[] Params;  // same layout as NeuralNetwork.GetVector()
    public readonly int MaxWidth;

    public FlatNetwork(NeuralNetwork net)
    {
        Shape = net.Shape;
        Params = net.GetVector();
        MaxWidth = Shape.Max();
    }
}
```

### 2. Scalar forward pass with caller-owned buffers

```csharp
// a and b have length >= MaxWidth. They can be [ThreadStatic] or owned by each vehicle.
public static ReadOnlySpan<double> Forward(FlatNetwork n, ReadOnlySpan<double> input,
                                           Span<double> a, Span<double> b)
{
    input.CopyTo(a);
    Span<double> cur = a, next = b;
    int p = 0;
    for (int l = 0; l < n.Shape.Length - 1; l++)
    {
        int ins = n.Shape[l], outs = n.Shape[l + 1];
        next[..outs].Clear();
        for (int i = 0; i < ins; i++)
        {
            double v = cur[i];
            ReadOnlySpan<double> row = n.Params.AsSpan(p + i * outs, outs);
            for (int j = 0; j < outs; j++) next[j] += v * row[j];
        }
        p += ins * outs;
        for (int j = 0; j < outs; j++) next[j] = Math.Tanh(next[j] + n.Params[p + j]);
        p += outs;
        Span<double> t = cur; cur = next; next = t;
    }
    return cur[..n.Shape[^1]];
}
```

Notes:

- Keep `Math.Tanh`. MathNet's `PointwiseTanh` calls it (✓). The Rust port has to reimplement it bit-for-bit (`game_tanh`); C# gets it for free.
- `next[j] += v * row[j]` must stay a separate multiply and add. ? RyuJIT does not contract `a + b * c` into an FMA on its own; only explicit `Math.FusedMultiplyAdd` or `Fma.MultiplyAdd` does. Confirm this with a bit-for-bit comparison test (below).
- ? This assumes MathNet uses its managed provider, which is the default when no native MKL or OpenBLAS provider is loaded. If the game ever enables a native provider, its summation order can differ, and the comparison test would catch that.
- The sensors can write straight into the input span, and the outputs can be read by index (finding 1). That removes the `NeuralNetworkInput` and `NeuralNetworkOutput` wrappers from the hot path.

### 3. Optional SIMD

With `System.Runtime.Intrinsics`, process 4 outputs per `Vector256<double>`. Keep `Avx.Multiply` and `Avx.Add` as separate calls:

```csharp
for (int j = 0; j + 4 <= outs; j += 4)
{
    var acc = Vector256<double>.Zero;
    for (int i = 0; i < ins; i++)
    {
        var w = Vector256.LoadUnsafe(ref row0, (nuint)(i * outs + j));
        acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(cur[i]), w));
    }
    acc.StoreUnsafe(ref next0, (nuint)j);
}
// Handle the remaining outs % 4 outputs with the scalar loop.
```

Each lane adds its terms in the same order as the scalar loop, so the result is identical. The Rust kernel also keeps up to 4 accumulators (16 outputs) in registers per pass over the inputs. That reduces how often `cur[i]` is broadcast; it is optional.

## How to check exactness

1. Load a saved population.
2. Feed the same random inputs (including edge values such as 0, ±1 and large values) through both the old `Forward` and the new one.
3. Compare the output arrays with `BitConverter.DoubleToInt64Bits` and require equality, not a tolerance.
4. Replay a recorded run with the new code. Positions after N ticks must match exactly.

## Expected gain

✓ The Rust port runs the same kind of network on the CPU at more than 1M car ticks per second on 2 threads, including physics and sensors (`reports/cpu-port-bench.json`). ? For the game, the gain depends on how much of each tick goes to inference and garbage collection. That has not been profiled. Removing the allocations alone should be clearly measurable with large populations, because it also reduces the GC pauses that stop every worker thread.
