
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

namespace Shorokoo.Core
{
    public interface IOptionalTensor : IVariable
    {
    }

    /// <summary>
    /// Immutable (class) graph node for an optional tensor — the value the graph actually stores
    /// (<see cref="Shorokoo.Core.Nodes.Node"/> outputs are <see cref="IVariable"/>). This type is
    /// deliberately minimal: the user-facing API lives on the value-type handle
    /// <see cref="OptionalTensor{T}"/>, which wraps one of these.
    /// </summary>
    public class ImmutableOptionalTensor<T> : Variable<T>, IOptionalTensor where T : IVarType
    {
        public ImmutableOptionalTensor(DType type, Node owningNode, Function? moduleFn, string? name) : base(type, owningNode, moduleFn, name) {}
    }

    /// <summary>
    /// Value-type handle for an optional tensor. The original <c>OptionalTensor&lt;T&gt;</c> name now
    /// denotes this <see langword="struct"/>; the reference type was renamed
    /// <see cref="ImmutableOptionalTensor{T}"/>. This struct carries the full user-facing surface.
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
        private ImmutableOptionalTensor<T>? inner;

        /// <summary>The wrapped immutable, materialising an absent optional for a defaulted handle.</summary>
        internal ImmutableOptionalTensor<T> Imm
            => inner ??= (ImmutableOptionalTensor<T>)OnnxOp.Optional(null, DataStructure.Tensor, OnnxUtils.GetDType<T>());

        // Wrap / unwrap between the handle and its immutable.
        public static implicit operator OptionalTensor<T>(ImmutableOptionalTensor<T> imm)
            => new OptionalTensor<T> { inner = imm };
        public static implicit operator ImmutableOptionalTensor<T>(OptionalTensor<T> handle)
            => handle.Imm;

        /// <summary>
        /// Implicitly unwraps an optional to a nullable tensor (<c>Tensor&lt;T&gt;?</c>) by reading
        /// its element. An absent handle (defaulted, <c>inner == null</c>) maps to <c>null</c>;
        /// otherwise the element is taken via <see cref="TensorValue"/>.
        /// </summary>
        public static implicit operator Tensor<T>?(OptionalTensor<T> optional)
            => optional.inner is null ? null : optional.TensorValue();

        // ── User-facing API (the optional surface lives here, not on the immutable) ──
        public IVariable Value() => OnnxOp.OptionalGetElement(Imm);
        public Tensor<T> TensorValue() => (Tensor<T>)Value();
        public Tensor<T> SequenceValue() => (Tensor<T>)Value();
        public Scalar<bit> HasValue() => (Scalar<bit>)OnnxOp.OptionalHasElement(Imm);

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
