
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Core.Utils;
using Shorokoo.Onnx;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;

namespace Shorokoo.Core
{
    public interface ITensorSequence : IVariable
    {
        public Scalar<int64> Count { get; }
        ITensor Concat(long axis, bool newAxis = false);
        ITensor this[Scalar<int64> index] { get; }
        ITensorSequence RemoveAt(Scalar<int64> index);
        ITensorSequence InsertAt(ITensor tensor, Scalar<int64> index);

        /// <summary>
        /// Inserts an arbitrary IVariable (e.g. an <see cref="ITensorStruct"/> for
        /// sequence-of-struct cases) at the given position, or appends when
        /// <paramref name="index"/> is null. Default implementation lowers directly to
        /// <c>SEQUENCE_INSERT</c>.
        /// </summary>
        public ITensorSequence Insert(IVariable variable, Scalar<int64>? index = null)
            => (ITensorSequence)OnnxOp.SequenceInsert(this, variable, index);

        public static ITensorSequence CreateEmpty(DType dtype)
                => (ITensorSequence)OnnxOp.SequenceEmpty(dtype);

        public static ITensorSequence Create(IVariable[] variables)
            => (ITensorSequence)OnnxOp.SequenceConstruct(variables);
    }

    /// <summary>
    /// Immutable (class) graph node for a tensor sequence — the value the graph stores
    /// (<see cref="Shorokoo.Core.Nodes.Node"/> outputs are <see cref="IVariable"/>). Minimal by
    /// design: it satisfies the <see cref="ITensorSequence"/> contract (so a graph value is
    /// recognised as a sequence) via explicit implementations and keeps only the internal
    /// graph-plumbing factories. The user-facing typed API lives on the value-type handle
    /// <see cref="TensorSequence{T}"/>.
    /// </summary>
    public class ImmutableTensorSequence<T> : Variable<T>, ITensorSequence
        where T : IVarType
    {
        internal ImmutableTensorSequence(DType dtype, Node owningNode, Function? moduleFn, string? name) : base(dtype, owningNode, moduleFn, name) {}

        Scalar<int64> ITensorSequence.Count => (Scalar<int64>)OnnxOp.SequenceLength(this);
        ITensor ITensorSequence.Concat(long axis, bool newAxis) => (Tensor<T>)OnnxOp.ConcatFromSequence(this, axis, newAxis);
        ITensor ITensorSequence.this[Scalar<int64> index] => (Tensor<T>)OnnxOp.SequenceAt(this, index);
        ITensorSequence ITensorSequence.RemoveAt(Scalar<int64> index) => (ImmutableTensorSequence<T>)OnnxOp.SequenceErase(this, index);
        ITensorSequence ITensorSequence.InsertAt(ITensor tensor, Scalar<int64> index) => (ImmutableTensorSequence<T>)OnnxOp.SequenceInsert(this, tensor, index);

        // Internal graph-plumbing factories (not user-facing surface).
        internal static ImmutableTensorSequence<T> CreateEmpty()
            => (ImmutableTensorSequence<T>)OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>());

        internal static ImmutableTensorSequence<T> CreateEmpty(Function targetFunction)
            => (ImmutableTensorSequence<T>)OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>(), targetFunction);

        internal static ImmutableTensorSequence<T> Create(Tensor<T>[] tensors)
            => tensors.Length == 0 ? CreateEmpty() :
                    (ImmutableTensorSequence<T>)OnnxOp.SequenceConstruct(tensors);

        internal static ImmutableTensorSequence<T> Create(Tensor<T>[] tensors, Function targetFunction)
            => tensors.Length == 0 ? CreateEmpty(targetFunction) :
                    (ImmutableTensorSequence<T>)OnnxOp.SequenceConstruct(targetFunction, tensors);
    }

    /// <summary>
    /// Value-type handle for a tensor sequence. The original <c>TensorSequence&lt;T&gt;</c> name now
    /// denotes this <see langword="struct"/>; the reference type was renamed
    /// <see cref="ImmutableTensorSequence{T}"/>. This struct carries the full user-facing typed API.
    /// The struct holds the immutable directly in a field (value-copy semantics for the Module DSL);
    /// a defaulted handle lazily materialises an empty sequence. This pass only makes mutation
    /// possible — behaviour is unchanged (de-facto immutable).
    /// </summary>
    public struct TensorSequence<T> : ITensorSequence where T : IVarType
    {
        private ImmutableTensorSequence<T>? inner;

        /// <summary>The wrapped immutable, materialising an empty sequence for a defaulted handle.</summary>
        internal ImmutableTensorSequence<T> Imm => inner ??= ImmutableTensorSequence<T>.CreateEmpty();

        public static implicit operator TensorSequence<T>(ImmutableTensorSequence<T> imm)
            => new TensorSequence<T> { inner = imm };
        public static implicit operator ImmutableTensorSequence<T>(TensorSequence<T> handle)
            => handle.Imm;

        // ── User-facing typed API (the sequence surface lives here, not on the immutable) ──
        public Scalar<int64> Count => (Scalar<int64>)OnnxOp.SequenceLength(Imm);

        public Tensor<T> Concat(long axis, bool newAxis = false)
            => (Tensor<T>)OnnxOp.ConcatFromSequence(Imm, axis, newAxis);

        public Tensor<T> this[Scalar<int64> index]
            => (Tensor<T>)OnnxOp.SequenceAt(Imm, index);

        /// <summary>Removes the element at <paramref name="index"/>, or the LAST element when called
        /// without an index (ONNX SequenceErase's optional-position default).</summary>
        public TensorSequence<T> RemoveAt(Scalar<int64>? index = null)
            => Imm.OwningNode.TargetFunction is null ?
                    (ImmutableTensorSequence<T>)OnnxOp.SequenceErase(Imm, index) :
                    (ImmutableTensorSequence<T>)OnnxOp.SequenceErase(Imm.OwningNode.TargetFunction, Imm, index);

        public TensorSequence<T> InsertAt(Tensor<T> tensor, Scalar<int64>? index)
            => Imm.OwningNode.TargetFunction is null ?
                    (ImmutableTensorSequence<T>)OnnxOp.SequenceInsert(Imm, tensor, index) :
                    (ImmutableTensorSequence<T>)OnnxOp.SequenceInsert(Imm.OwningNode.TargetFunction, Imm, tensor, index);

        public TensorSequence<T> Append(Tensor<T> tensor) => this.InsertAt(tensor, null);

        public static TensorSequence<T> CreateEmpty() => ImmutableTensorSequence<T>.CreateEmpty();
        internal static TensorSequence<T> CreateEmpty(Function targetFunction) => ImmutableTensorSequence<T>.CreateEmpty(targetFunction);
        public static TensorSequence<T> Create(Tensor<T>[] tensors) => ImmutableTensorSequence<T>.Create(tensors);
        internal static TensorSequence<T> Create(Tensor<T>[] tensors, Function targetFunction) => ImmutableTensorSequence<T>.Create(tensors, targetFunction);

        // ITensorSequence explicit members (interface signatures, returning interface types).
        Scalar<int64> ITensorSequence.Count => this.Count;
        ITensor ITensorSequence.Concat(long axis, bool newAxis) => this.Concat(axis, newAxis);
        ITensor ITensorSequence.this[Scalar<int64> index] => this[index];
        ITensorSequence ITensorSequence.RemoveAt(Scalar<int64> index) => this.RemoveAt(index);
        ITensorSequence ITensorSequence.InsertAt(ITensor tensor, Scalar<int64> index) => this.InsertAt((Tensor<T>)tensor, index);

        // IVariable surface — forward to the wrapped immutable.
        public Node OwningNode => Imm.OwningNode;
        public DType Type => Imm.Type;
        public Function? ModuleFn => Imm.ModuleFn;
        public TensorKey Key => Imm.Key;
        public string UniqueName => Imm.UniqueName;
        public bool IsValid { get => Imm.IsValid; set => Imm.IsValid = value; }
        public Variable<V> As<V>() where V : IVarType => ((IVariable)Imm).As<V>();

#pragma warning disable CS0618 // forwarding the obsolete member is intentional
        string? IVariable.FriendlyName => ((IVariable)Imm).FriendlyName;
#pragma warning restore CS0618
    }
}
