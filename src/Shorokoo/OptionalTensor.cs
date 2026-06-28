
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using Shorokoo.Core;
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
using Shorokoo;

namespace Shorokoo
{
    public interface IOptionalTensor : IValue
    {
    }

    /// <summary>
    /// Value-type handle for an optional tensor. The original <c>OptionalTensor&lt;T&gt;</c> name now
    /// denotes this <see langword="struct"/>; the reference type was renamed
    /// <see cref="Variable"/>. This struct carries the full user-facing surface.
    /// <para>
    /// The struct holds the immutable <b>directly</b> in a field (no shared box) so copying a handle
    /// copies the reference field by value — giving the Module DSL value-type semantics: a callee
    /// mutating its parameter does not affect the caller. This pass only makes mutation
    /// <i>possible</i>; nothing mutates yet, so behaviour is unchanged (de-facto immutable).
    /// </para>
    /// <para>
    /// A zero-initialised handle (<c>default</c>, <c>inner == null</c>) lazily materialises an
    /// <b>absent</b> optional on first use.
    /// </para>
    /// </summary>
    public struct OptionalTensor<T> : IOptionalTensor where T : IVarType
    {
        private Variable? inner;

        Variable? IValue.Immutable => Imm;

        /// <summary>The wrapped immutable, materialising an absent optional for a defaulted handle.</summary>
        internal Variable Imm
            => inner ??= OnnxOp.Optional(null, DataStructure.Tensor, OnnxUtils.GetDType<T>());

        // Wrap / unwrap between the handle and its immutable.
        private static readonly DType? expectedDType = OnnxUtils.GetDType(typeof(T));
        public static implicit operator OptionalTensor<T>(Variable imm)
        {
            IValue.RequireKind(imm, DataStructure.Optional);
            IValue.RequireDType(imm, expectedDType);
            return new OptionalTensor<T> { inner = imm };
        }
        public static implicit operator Variable(OptionalTensor<T> handle)
            => handle.Imm;

        /// <summary>
        /// Implicitly unwraps an optional to a nullable tensor (<c>Tensor&lt;T&gt;?</c>) by reading
        /// its element. An absent handle (defaulted, <c>inner == null</c>) maps to <c>null</c>;
        /// otherwise the element is taken via <see cref="TensorValue"/>.
        /// </summary>
        public static implicit operator Tensor<T>?(OptionalTensor<T> optional)
            => optional.inner is null ? default(Tensor<T>?) : optional.TensorValue();

        // ── User-facing API (the optional surface lives here, not on the immutable) ──
        public Variable Value() => OnnxOp.OptionalGetElement(Imm);
        public Tensor<T> TensorValue() => (Variable)Value();
        public Tensor<T> SequenceValue() => (Variable)Value();
        public Scalar<bit> HasValue() => OnnxOp.OptionalHasElement(Imm);

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
