using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static Shorokoo.Globals;

namespace Shorokoo
{
    /// <summary>
    /// One axis of a <see cref="Tensor{T}"/> multi-axis indexer. Implicitly built from a
    /// <see cref="Range"/> (<c>1..3</c>), a stepped range (<c>(1..3, 2L)</c>), explicit
    /// (from, to, step) bounds, a single index (<c>0L</c> / <c>^1</c>), or a <see cref="Tensor{T}"/>
    /// of gather indices.
    ///
    /// <para>PyTorch <c>t[3:8]</c> -> Shorokoo <c>t[3..8]</c>; <c>t[::2]</c> -> <c>t[(.., 2L)]</c>;
    /// <c>t[-2]</c> -> <c>t[^2]</c>.</para>
    /// </summary>
    public struct TensorIndexerParam
    {
        internal readonly bool IsIndex { get; init; }
        internal readonly bool IsFullRange { get; init; }
        internal readonly Tensor<int64>? Indices { get; init; }
        internal readonly Scalar<int64>? ScalarStart { get; init; }
        internal readonly Scalar<int64>? ScalarEnd { get; init; }
        internal readonly Scalar<int64>? ScalarStep { get; init; }

        public static implicit operator TensorIndexerParam(Range range)
            => new TensorIndexerParam
            {
                ScalarStart = Scalar(OnnxUtils.FromIndex(range.Start)),
                ScalarEnd = Scalar(OnnxUtils.FromIndex(range.End)),
                ScalarStep = null,   // step 1
                IsIndex = false,
                IsFullRange = range.Equals(Range.All)
            };

        public static implicit operator TensorIndexerParam((Range range, long step) tuple)
            => new TensorIndexerParam
            {
                ScalarStart = Scalar(OnnxUtils.FromIndex(tuple.range.Start)),
                ScalarEnd = Scalar(OnnxUtils.FromIndex(tuple.range.End)),
                ScalarStep = tuple.step == 1 ? (Scalar<int64>?)null : Scalar(tuple.step),
                IsIndex = false,
                IsFullRange = tuple.step == 1 && tuple.range.Equals(Range.All)
            };

        public static implicit operator TensorIndexerParam((long from, long? to, long step) tuple)
            => new TensorIndexerParam
            {
                ScalarStart = Scalar(tuple.from),
                ScalarEnd = Scalar(tuple.to ?? long.MaxValue),
                ScalarStep = tuple.step == 1 ? (Scalar<int64>?)null : Scalar(tuple.step),
                IsIndex = false,
                IsFullRange = tuple.from == 0 && (tuple.to ?? long.MaxValue) == long.MaxValue
            };

        public static implicit operator TensorIndexerParam(long index)
            => new TensorIndexerParam
            {
                ScalarStart = Scalar(index),
                ScalarEnd = Scalar(index == -1 ? long.MaxValue : index + 1),
                ScalarStep = null,   // single element (size-1, step 1)
                IsIndex = true
            };

        public static implicit operator TensorIndexerParam(Index index)
            => (TensorIndexerParam)OnnxUtils.FromIndex(index);

        public static implicit operator TensorIndexerParam(Tensor<int64> indices)
            => new TensorIndexerParam { Indices = indices };

        public static implicit operator TensorIndexerParam(Vector<int64> indices)
            => new TensorIndexerParam { Indices = indices };

        public static implicit operator TensorIndexerParam(long[] indices)
            => new TensorIndexerParam { Indices = Vector(indices) };
    }

    public partial struct Tensor<T>
    {
        /// <summary>
        /// Slices/gathers the tensor for reads (<c>Tensor r = t[1..3, 0..2];</c>), and replaces the
        /// indexed region for writes (<c>t[1..3, 0..2] = r;</c>). One <see cref="TensorIndexerParam"/>
        /// per leading axis; omitted trailing axes are taken in full. Because <see cref="Tensor{T}"/>
        /// is a value type, an indexer-set rebinds this local handle and never mutates a caller's instance.
        /// </summary>
        public Tensor<T> this[params TensorIndexerParam[] slices]
        {
            get => TensorIndexer.Read(this, slices);
            set => this = TensorIndexer.Write(this, slices, value);
        }
    }

    /// <summary>Read/write implementations behind the <see cref="Tensor{T}"/> multi-axis indexer.</summary>
    internal static class TensorIndexer
    {
        public static Tensor<TT> Read<TT>(Tensor<TT> source, TensorIndexerParam[] slices) where TT : IVarType
        {
            Debug.Assert(slices.Length > 0);

            var gathers = slices.Where(x => x.Indices is not null).ToArray();
            if (gathers.Length > 1)
                throw new InvalidTensorOperationException(ErrorCodes.TIH004, "GatherND", $"{gathers.Length} index tensors",
                    "At most one indexer axis may be a gather-index tensor.");

            var result = ApplySlices(source, slices, out var leadingGatherAxis);

            // A single gather-index axis: gather along its position (after any leading slices).
            if (gathers.Length == 1)
            {
                var indices = gathers[0].Indices!.Value;
                result = result.Gather(indices, axis: leadingGatherAxis);
            }

            return result;
        }

        public static Tensor<TT> Write<TT>(Tensor<TT> source, TensorIndexerParam[] slices, Tensor<TT> values) where TT : IVarType
        {
            var gathers = slices.Where(x => x.Indices is not null).ToArray();
            if (gathers.Length > 1)
                throw new InvalidTensorOperationException(ErrorCodes.TIH006, "ScatterND", $"{gathers.Length} index tensors",
                    "At most one indexer axis may be a gather-index tensor.");

            bool hasSlice = slices.Any(x => x.Indices is null && !x.IsFullRange);

            // Pure gather-index assignment on axis 0: scatter whole rows.
            if (gathers.Length == 1 && !hasSlice)
            {
                var idx = ((Tensor<int64>)gathers[0].Indices!.Value).Unsqueeze();   // [n] -> [n, 1]
                return source.ScatterND(idx, values);
            }

            // Slice/index assignment: place `values` into the selected region and select with Where.
            return WriteRegion(source, slices, values);
        }

        /// <summary>Applies the slice/single-index axes (full-range axes untouched), squeezing single-index axes.</summary>
        private static Tensor<TT> ApplySlices<TT>(Tensor<TT> source, TensorIndexerParam[] slices, out long leadingGatherAxis) where TT : IVarType
        {
            var axes = new List<Scalar<int64>>();
            var starts = new List<Scalar<int64>>();
            var ends = new List<Scalar<int64>>();
            var steps = new List<Scalar<int64>>();
            var squeeze = new List<Scalar<int64>>();
            leadingGatherAxis = 0;

            for (long i = 0; i < slices.Length; i++)
            {
                var s = slices[i];
                if (s.Indices is not null) { leadingGatherAxis = i; continue; }
                if (s.IsFullRange) continue;
                axes.Add(Scalar(i));
                starts.Add(s.ScalarStart!.Value);
                ends.Add(s.ScalarEnd!.Value);
                steps.Add(s.ScalarStep ?? Scalar(1L));
                if (s.IsIndex) squeeze.Add(Scalar(i));
            }

            var result = source;
            if (axes.Count > 0)
            {
                Vector<int64> startsV = [.. starts];
                Vector<int64> endsV = [.. ends];
                Vector<int64> axesV = [.. axes];
                Vector<int64> stepsV = [.. steps];
                result = result.Slice(startsV, endsV, axesV, stepsV);
                if (squeeze.Count > 0)
                    result = result.Squeeze([.. squeeze]);
            }
            return result;
        }

        /// <summary>Replaces the (contiguous) slice/index region with <paramref name="values"/> via a mask + Where.</summary>
        private static Tensor<TT> WriteRegion<TT>(Tensor<TT> source, TensorIndexerParam[] slices, Tensor<TT> values) where TT : IVarType
        {
            var shape = source.TShape;

            // Only the constrained (slice/single-index) axes are padded; full-range and unspecified
            // trailing axes already span the source, so we drive ONNX Pad by `axes` and never need
            // the source's static rank.
            var padAxes = new List<Scalar<int64>>();
            var left = new List<Scalar<int64>>();
            var right = new List<Scalar<int64>>();
            var unsqueezeAxes = new List<Scalar<int64>>();
            for (long i = 0; i < slices.Length; i++)
            {
                var s = slices[i];
                if (s.Indices is not null || s.IsFullRange)
                    continue;
                if (s.ScalarStep is not null)
                    throw new InvalidTensorOperationException(ErrorCodes.TIH003, "Step slice assignment", "strided tensor assignment",
                        "Strided slice assignment on a multi-axis tensor is not supported.");
                var dim = shape[i];
                var start = s.ScalarStart!.Value;
                var end = s.ScalarEnd!.Value.Min(dim);
                padAxes.Add(Scalar(i));
                left.Add(start);
                right.Add(dim - end);
                if (s.IsIndex) unsqueezeAxes.Add(Scalar(i));   // values dropped this axis; restore it (size 1)
            }

            // Restore axes the matching read would have squeezed so `values` matches the region rank.
            var placed = values;
            if (unsqueezeAxes.Count > 0)
                placed = placed.Unsqueeze([.. unsqueezeAxes]);

            if (padAxes.Count == 0)
                return placed;   // every axis full -> the values replace the whole tensor

            Vector<int64> axesV = [.. padAxes];
            Vector<int64> leftV = [.. left];
            Vector<int64> rightV = [.. right];

            var region = placed.Pad(PadMode.Constant, leftV, rightV, null, axesV);
            var mask = Tensor<bit>.Fill(placed.TShape, Shorokoo.Core.InternalGlobals.OnnxTensorData(1, true))
                .Pad(PadMode.Constant, leftV, rightV, null, axesV);
            return mask.Where(region, source);
        }
    }
}
