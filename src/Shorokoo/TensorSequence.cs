
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

namespace Shorokoo
{
    public interface ITensorSequence : IValue
    {
        public Scalar<int64> Count { get; }
        ITensor Concat(long axis, bool newAxis = false);
        ITensor this[Scalar<int64> index] { get; }
        ITensorSequence RemoveAt(Scalar<int64> index);
        ITensorSequence InsertAt(ITensor tensor, Scalar<int64> index);
    }

    /// <summary>
    /// Value-type handle for a tensor sequence. The original <c>TensorSequence&lt;T&gt;</c> name now
    /// denotes this <see langword="struct"/>; the reference type was renamed
    /// <see cref="Variable"/>. This struct carries the full user-facing typed API.
    /// The struct holds the immutable directly in a field (value-copy semantics for the Module DSL);
    /// a defaulted handle lazily materialises an empty sequence. This pass only makes mutation
    /// possible — behaviour is unchanged (de-facto immutable).
    /// </summary>
    public struct TensorSequence<T> : ITensorSequence where T : IVarType
    {
        private Variable? inner;

        Variable? IValue.Immutable => Imm;

        /// <summary>The wrapped immutable, materialising an empty sequence for a defaulted handle.</summary>
        internal Variable Imm => inner ??= OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>());

        private static readonly DType? expectedDType = OnnxUtils.GetDType(typeof(T));
        public static implicit operator TensorSequence<T>(Variable imm)
        {
            IValue.RequireKind(imm, DataStructure.Sequence);
            IValue.RequireDType(imm, expectedDType);
            return new TensorSequence<T> { inner = imm };
        }
        public static implicit operator Variable(TensorSequence<T> handle)
            => handle.Imm;

        // ── User-facing typed API (the sequence surface lives here, not on the immutable) ──
        public Scalar<int64> Count => OnnxOp.SequenceLength(Imm);

        public Tensor<T> Concat(long axis, bool newAxis = false)
            => OnnxOp.ConcatFromSequence(Imm, axis, newAxis);

        public Tensor<T> this[Scalar<int64> index]
            => OnnxOp.SequenceAt(Imm, index);

        /// <summary>Removes the element at <paramref name="index"/>, or the LAST element when called
        /// without an index (ONNX SequenceErase's optional-position default).</summary>
        public TensorSequence<T> RemoveAt(Scalar<int64>? index = null)
            => Imm.OwningNode.TargetFunction is null ?
                    OnnxOp.SequenceErase(Imm, index) :
                    OnnxOp.SequenceErase(Imm.OwningNode.TargetFunction, Imm, index);

        public TensorSequence<T> InsertAt(Tensor<T> tensor, Scalar<int64>? index)
            => Imm.OwningNode.TargetFunction is null ?
                    OnnxOp.SequenceInsert(Imm, tensor, index) :
                    OnnxOp.SequenceInsert(Imm.OwningNode.TargetFunction, Imm, tensor, index);

        public TensorSequence<T> Append(Tensor<T> tensor) => this.InsertAt(tensor, null);

        // Typed factories live here on the handle (the element type T is known here, not on the node).
        public static TensorSequence<T> CreateEmpty()
            => OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>());
        internal static TensorSequence<T> CreateEmpty(Function targetFunction)
            => OnnxOp.SequenceEmpty(OnnxUtils.GetDType<T>(), targetFunction);
        public static TensorSequence<T> Create(Tensor<T>[] tensors)
            => tensors.Length == 0 ? CreateEmpty()
                : OnnxOp.SequenceConstruct([.. tensors.Select(t => (Variable)t)]);
        internal static TensorSequence<T> Create(Tensor<T>[] tensors, Function targetFunction)
            => tensors.Length == 0 ? CreateEmpty(targetFunction)
                : OnnxOp.SequenceConstruct(targetFunction, [.. tensors.Select(t => (Variable)t)]);

        // ITensorSequence explicit members (interface signatures, returning interface types).
        Scalar<int64> ITensorSequence.Count => this.Count;
        ITensor ITensorSequence.Concat(long axis, bool newAxis) => this.Concat(axis, newAxis);
        ITensor ITensorSequence.this[Scalar<int64> index] => this[index];
        ITensorSequence ITensorSequence.RemoveAt(Scalar<int64> index) => this.RemoveAt(index);
        ITensorSequence ITensorSequence.InsertAt(ITensor tensor, Scalar<int64> index) => this.InsertAt((Tensor<T>)tensor, index);

        // IValue surface — forward to the wrapped immutable.
        public Node OwningNode => Imm.OwningNode;
        public DType Type => Imm.Type;
        public Function? ModuleFn => Imm.ModuleFn;
        public TensorKey Key => Imm.Key;
        public string UniqueName => Imm.UniqueName;
        public bool IsValid { get => Imm.IsValid; set => Imm.IsValid = value; }
        public Tensor<V> As<V>() where V : IVarType => Tensor<V>.Reinterpret(Imm);

#pragma warning disable CS0618 // forwarding the obsolete member is intentional
        string? IValue.FriendlyName => ((IValue)Imm).FriendlyName;
#pragma warning restore CS0618
    }
}
