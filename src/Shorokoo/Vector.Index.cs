
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Net.Mail;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static Shorokoo.Globals;
using static Shorokoo.Core.InternalGlobals;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo
{
    /// <summary>
    /// Argument type for the <see cref="Vector{T}"/> slicing indexer. Implicitly built from a
    /// <see cref="Range"/>, a (range, step) or (from, to, step) tuple, an index vector, or a
    /// <c>long[]</c> of gather indices.
    /// </summary>
    public struct VectorIndexerParam
    {
        internal readonly Vector<int64>? Indices { get; init; }
        internal readonly bool IsFullRange { get; init; }
        internal readonly Scalar<int64>? ScalarStart { get; init; }
        internal readonly Scalar<int64>? ScalarEnd { get; init; }
        internal readonly Scalar<int64>? ScalarStep { get; init; }

        /// <summary>Wraps a <see cref="Range"/> as a slicing parameter (step 1).</summary>
        public static implicit operator VectorIndexerParam(Range range)
            => new VectorIndexerParam
            {
                ScalarStart = Scalar(OnnxUtils.FromIndex(range.Start)),
                ScalarEnd = Scalar(OnnxUtils.FromIndex(range.End)),
                ScalarStep = null, // Scalar(1L),
                IsFullRange = range.Equals(Range.All)
            };

        /// <summary>Wraps a <see cref="Range"/> plus an explicit step.</summary>
        public static implicit operator VectorIndexerParam((Range range, long step) tuple)
            => new VectorIndexerParam
            {
                ScalarStart = Scalar(OnnxUtils.FromIndex(tuple.range.Start)),
                ScalarEnd = Scalar(OnnxUtils.FromIndex(tuple.range.End)),
                ScalarStep = Scalar(tuple.step),
                IsFullRange = tuple.step == 1 && tuple.range.Equals(Range.All)
            };

        /// <summary>Wraps explicit from / to / step slice bounds (null <c>to</c> means to-the-end).</summary>
        public static implicit operator VectorIndexerParam((long from, long? to, long step) tuple)
            => new VectorIndexerParam
            {
                ScalarStart = Scalar(tuple.from),
                ScalarEnd = Scalar(tuple.to ?? long.MaxValue),
                ScalarStep = Scalar(tuple.step),
                IsFullRange = tuple.from == 0 && (tuple.to ?? long.MaxValue) == long.MaxValue
            };

        /// <summary>Wraps a runtime index vector — the indexer gathers those positions.</summary>
        public static implicit operator VectorIndexerParam(Vector<int64> indices)
            => new VectorIndexerParam { Indices = indices };

        /// <summary>Wraps constant gather indices.</summary>
        public static implicit operator VectorIndexerParam(long[] indices)
            => new VectorIndexerParam { Indices = Vector(indices) };
    }

    public partial struct Vector<T>
    {
        /// <summary>
        /// Slices or gathers the vector for reads (<c>Vector w = v[1..3];</c>), and replaces the
        /// indexed positions for writes (<c>v[1..3] = w;</c>). Because <see cref="Vector{T}"/> is a
        /// value type, an indexer-set rebinds this local handle and never mutates a caller's instance.
        /// </summary>
        public Vector<T> this[VectorIndexerParam index]
        {
            get => VectorIndexer.ReadSlice(this, index);
            set => this = VectorIndexer.WriteSlice(this, index, value);
        }

        /// <summary>Reads or writes a single element at a runtime index.</summary>
        public Scalar<T> this[Scalar<int64> index]
        {
            get => this.Gather(index.Unsqueeze(), 0).Squeeze().Scalar();
            set => this = this.ScatterND(index.Unsqueeze().Unsqueeze(), value.Unsqueeze());
        }

        /// <summary>Reads or writes a single element at a constant index.</summary>
        public Scalar<T> this[long index]
        {
            get => this[Globals.Scalar(index)];
            set => this[Globals.Scalar(index)] = value;
        }

        /// <summary>Reads or writes a single element at a constant index.</summary>
        public Scalar<T> this[int index]
        {
            get => this[(long)index];
            set => this[(long)index] = value;
        }

        /// <summary>Reads or writes a single element at an <see cref="Index"/> (supports from-end <c>^i</c>).</summary>
        public Scalar<T> this[Index index]
        {
            get => this[OnnxUtils.FromIndex(index)];
            set => this[OnnxUtils.FromIndex(index)] = value;
        }
    }

    /// <summary>
    /// Read/write implementations behind the <see cref="Vector{T}"/> slicing indexer. Reads
    /// materialise the sliced/gathered vector; writes return a copy of the source with the
    /// indexed positions replaced (the indexer setter rebinds the handle to it).
    /// </summary>
    internal static class VectorIndexer
    {
        public static Vector<TT> ReadSlice<TT>(Vector<TT> source, VectorIndexerParam slice) where TT : IVarType
        {
            // Full range is the identity.
            if (slice.IsFullRange)
                return source;

            // Gather a list of positions (ONNX Gather along axis 0 keeps rank 1).
            if (slice.Indices is not null)
                return source.Gather(slice.Indices.Value, 0).Vec();

            Debug.Assert(slice.ScalarStart is not null);
            Debug.Assert(slice.ScalarEnd is not null);

            // Contiguous or strided slice (ORT clamps an open-ended bound; steps drive the stride).
            return source.Slice(slice.ScalarStart!.Value, slice.ScalarEnd!.Value, slice.ScalarStep);
        }

        public static Vector<TT> WriteSlice<TT>(Vector<TT> source, VectorIndexerParam slice, Vector<TT> values) where TT : IVarType
        {
            // Full range replaces the whole vector.
            if (slice.IsFullRange)
                return values;

            // Scatter the values onto the targeted positions. Both the gather-by-index and the
            // (possibly strided) slice cases reduce to "scatter `values` at these row indices":
            // ScatterND wants the index list shaped [n, 1].
            Vector<int64> positions = slice.Indices ?? SlicePositions(source, slice);
            Tensor<int64> indices = ((Tensor<int64>)positions).Unsqueeze();
            return source.ScatterND(indices, values);
        }

        /// <summary>The concrete row indices a (possibly strided, possibly open-ended) slice selects.</summary>
        private static Vector<int64> SlicePositions<TT>(Vector<TT> source, VectorIndexerParam slice) where TT : IVarType
        {
            Debug.Assert(slice.ScalarStart is not null);
            Debug.Assert(slice.ScalarEnd is not null);

            var length = source.TShape[0];
            var start = slice.ScalarStart!.Value;
            var end = slice.ScalarEnd!.Value.Min(length);   // clamp an open-ended (long.MaxValue) bound
            var step = slice.ScalarStep ?? Scalar(1L);
            return OnnxOp.Range(start, end, step);
        }
    }
}
