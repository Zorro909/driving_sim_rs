// New file, e.g. AILearnsToDrive.Lib.Tracks/PathQueryCache.cs.

using System;
using Godot;

namespace AILearnsToDrive.Lib.Tracks;

/// <summary>
/// Remembers the last path queries on each thread. CorrectDirectionSensor and TrackCurvatureSensor
/// run back to back for the same car on the same thread and ask for the same closest offset and
/// the same transform, so the second sensor gets the first one's results instead of asking
/// Godot again. Keys are the curve instance and the exact float bits, so a hit returns exactly
/// what Godot returned for the same question. Assumes a curve's points are not edited in place
/// while it is in use (a new Curve2D is made per track); call Clear() if that ever changes.
/// </summary>
public static class PathQueryCache
{
    private sealed class State
    {
        public Curve2D? OffsetCurve;
        public int PointX, PointY;
        public float Offset;
        public Curve2D? SampleCurve0, SampleCurve1;
        public int SampleOffset0, SampleOffset1;
        public Transform2D Sample0, Sample1;
        public bool ReplaceFirst = true;
    }

    [ThreadStatic] private static State? _state;

    /// <summary>Same as curve.GetClosestOffset(point).</summary>
    public static float GetClosestOffset(Curve2D curve, Vector2 point)
    {
        State s = _state ??= new State();
        int x = BitConverter.SingleToInt32Bits(point.X), y = BitConverter.SingleToInt32Bits(point.Y);
        if (ReferenceEquals(s.OffsetCurve, curve) && s.PointX == x && s.PointY == y)
        {
            Verify(BitConverter.SingleToInt32Bits(s.Offset) == BitConverter.SingleToInt32Bits(curve.GetClosestOffset(point)), "GetClosestOffset");
            return s.Offset;
        }
        s.Offset = curve.GetClosestOffset(point);
        s.OffsetCurve = curve;
        s.PointX = x;
        s.PointY = y;
        return s.Offset;
    }

    /// <summary>Same as curve.SampleBakedWithRotation(offset). Keeps two entries: the curvature
    /// sensor samples at the closest offset and at a lookahead offset.</summary>
    public static Transform2D SampleBakedWithRotation(Curve2D curve, float offset)
    {
        State s = _state ??= new State();
        int bits = BitConverter.SingleToInt32Bits(offset);
        if (ReferenceEquals(s.SampleCurve0, curve) && s.SampleOffset0 == bits)
        {
            Verify(s.Sample0 == curve.SampleBakedWithRotation(offset), "SampleBakedWithRotation");
            return s.Sample0;
        }
        if (ReferenceEquals(s.SampleCurve1, curve) && s.SampleOffset1 == bits)
        {
            Verify(s.Sample1 == curve.SampleBakedWithRotation(offset), "SampleBakedWithRotation");
            return s.Sample1;
        }
        Transform2D result = curve.SampleBakedWithRotation(offset);
        if (s.ReplaceFirst)
        {
            (s.SampleCurve0, s.SampleOffset0, s.Sample0) = (curve, bits, result);
        }
        else
        {
            (s.SampleCurve1, s.SampleOffset1, s.Sample1) = (curve, bits, result);
        }
        s.ReplaceFirst = !s.ReplaceFirst;
        return result;
    }

    /// <summary>Forgets the cached queries of the calling thread.</summary>
    public static void Clear() => _state = null;

    // Define PATH_QUERY_CACHE_VERIFY to recompute every hit with Godot and report differences.
    [System.Diagnostics.Conditional("PATH_QUERY_CACHE_VERIFY")]
    private static void Verify(bool same, string query)
    {
        if (!same)
        {
            GD.PushError($"PathQueryCache: cached {query} differs from Godot's result.");
        }
    }
}
