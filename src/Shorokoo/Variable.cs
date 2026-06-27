
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
using System.Diagnostics;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static RandN.Distributions.Uniform;
using Shorokoo.Core.Nodes.NodeDefinitions;

#pragma warning disable CS8981

namespace Shorokoo
{
    public interface IVarType;

    public interface SimpleFloatLike : FloatLike, SimpleNumLike, SimpleNumLike2;
    public interface SimpleNumLike : AnyNumLike;
    public interface SimpleNumLike2 : AnyNumLike;

    public interface bfloat16 : FloatLike;
    public interface float16 : FloatLike;
    public interface float32 : SimpleFloatLike;
    public interface float64 : SimpleFloatLike;
    public interface int4 : AnySignedIntLike;
    public interface int8 : SignedIntLike, Int8Like;
    public interface int16 : SignedIntLike, SimpleNumLike;
    public interface int32 : IndexLike, SimpleNumLike, SimpleNumLike2;
    public interface int64 : IndexLike, SimpleNumLike, SimpleNumLike2;
    public interface uint4 : AnyUnsignedIntLike;
    public interface uint8 : UnsignedIntLike, Int8Like;
    public interface uint16 : UnsignedIntLike;
    public interface uint32 : UnsignedIntLike, SimpleNumLike2;
    public interface uint64 : UnsignedIntLike, SimpleNumLike2;
    // Variable-length UTF-8 string tensor element. Maps to ONNX
    // TensorProto.DataType.STRING (8). Element-of, not array-of: a Tensor<@string>
    // is a tensor whose individual elements are .NET strings.
    public interface @string : IVarType;
    public interface bit : CommonLike;
    public interface complex64 : ComplexLike;
    public interface complex128 : ComplexLike;
    public interface invalid : IVarType;

    public interface Int8Like : IntLike;
    public interface IndexLike : SignedIntLike;
    public interface SignedIntLike : AnySignedIntLike, IntLike;
    public interface UnsignedIntLike : AnyUnsignedIntLike, IntLike;
    public interface IntLike : AnyIntLike, NumLike;
    public interface AnyUnsignedIntLike : AnyUnsignedNumLike, AnyIntLike;
    public interface AnySignedIntLike : AnySignedNumLike, AnyIntLike;
    public interface AnyIntLike : AnyNumLike;

    public interface ComplexLike : AnySignedNumLike;
    public interface FloatLike : SignedNumLike;

    public interface IModuleVarType : ParamLike;
    public interface IModelVarType : ParamLike;

    // Generic type placeholders - used during VirtualGraph construction for unresolved generic type parameters
    // These derive from all primitive type interfaces so operators will accept them as valid
    public interface IGenericType : 
        IVarType,
        FloatLike,
        SignedIntLike,
        UnsignedIntLike,
        IntLike,
        NumLike,
        AnyNumLike,
        AnyLike,
        CommonLike,
        ParamLike
    { }

    // Specific generic type markers (one per generic type parameter: T, Q, R, etc.)
    public interface IGenericType1 : IGenericType { }
    public interface IGenericType2 : IGenericType { }
    public interface IGenericType3 : IGenericType { }
    public interface IGenericType4 : IGenericType { }
    public interface IGenericType5 : IGenericType { }
    public interface IGenericType6 : IGenericType { }
    public interface IGenericType7 : IGenericType { }
    public interface IGenericType8 : IGenericType { }

    public interface UnsignedNumLike : NumLike;
    public interface SignedNumLike : AnySignedNumLike, NumLike;
    public interface AnySignedNumLike : AnyNumLike;
    public interface AnyUnsignedNumLike : AnyNumLike;
    public interface AnyNumLike : ParamLike;
    public interface NumLike : AnyNumLike, CommonLike;
    public interface CommonLike : AnyLike;
    public interface AnyLike : ParamLike;
    public interface ParamLike : IVarType;

    /// <summary>
    /// Marker interface for TensorStruct types. All user-defined struct interfaces must derive from IStruct.
    /// IStruct has no associated DType - it's an abstract category marker like FloatLike or NumLike.
    /// </summary>
    public interface IStruct : IVarType;

    /// <summary>
    /// Built-in marker interface for TensorStruct types where the struct definition is not known at C# compile time.
    /// When using TensorStruct&lt;DTypeStruct&gt;, the struct definition comes from the DType at runtime rather than from interface property declarations.
    /// </summary>
    public interface DTypeStruct : IStruct;

    public interface IValue<out T> : IValue where T : IVarType;

    /// <summary>
    /// Non-generic immutable graph-value node — the object the Node/Variable graph is built from.
    /// Holds only runtime node state: the owning <see cref="Node"/>, the runtime element
    /// <see cref="DType"/>, the <see cref="TensorKey"/>, the producing module function, validity,
    /// and the serialization name. The element type lives in <see cref="Type"/> at runtime, NOT in
    /// a C# type parameter, and the structural kind (tensor / sequence / optional / struct) lives on
    /// the producing node — so a node needs neither generics nor a per-kind subclass.
    /// <para>
    /// Deliberately does <b>not</b> implement <see cref="IValue"/>: <c>IValue</c> is the user-facing
    /// handle interface (implemented by the value-struct handles such as <see cref="Tensor{T}"/>),
    /// and a graph node is not a handle. Generic typed reinterprets live on the
    /// <see cref="ImmutableVariable{T}"/> subclass during the migration; the end state collapses the
    /// generic subclasses into this one class.
    /// </para>
    /// </summary>
    public abstract class Variable : IValue
    {
        public Node OwningNode { get; private set; }

        public DType Type { get; private set; }

        public Function? ModuleFn { get; private set; }

        public bool IsValid { get; set; } = true;

        /// <summary>
        /// A globally unique identifier for this tensor, composed of the parent node's key and the output index.
        /// Set by the Node constructor after creating outputs.
        /// </summary>
        public TensorKey Key { get; private set; }

        private string? uniqueName;

        protected Variable(DType type, Node owningNode, Function? moduleFn, string? name)
        {
            this.OwningNode = owningNode;
            this.Type = type;
            this.uniqueName = name;  // Store the provided name as uniqueName
            this.ModuleFn = moduleFn;
        }

        /// <summary>
        /// Sets the TensorKey for this variable. Called by the Node constructor after creating outputs.
        /// </summary>
        internal void SetKey(TensorKey key)
        {
            this.Key = key;
        }

        /// <summary>
        /// Override this variable's UniqueName. Used by
        /// <see cref="Shorokoo.Graph.FastComputationGraphConverter.ToComputationGraph"/>
        /// to restore original graph-input/output names after a Fast↔CG roundtrip.
        /// </summary>
        internal void SetUniqueName(string? name)
        {
            this.uniqueName = name;
        }

        /// <summary>
        /// The unique name for this tensor. Defaults to Key.ToString() but can be set to human-readable
        /// names like "N1_T0" by processors during construction. Used for ONNX serialization.
        /// </summary>
        public string UniqueName => this.uniqueName ?? this.Key.ToString();

        /// <summary>
        /// Obsolete: Use UniqueName instead. DefaultName now redirects to UniqueName for backwards compatibility.
        /// </summary>
        [Obsolete("Use UniqueName instead. DefaultName is deprecated and will be removed in a future version.")]
        public string DefaultName => this.UniqueName;

        /// <summary>
        /// Deprecated: FriendlyName is no longer used. Use UniqueName for ONNX names or Key for stable identifiers.
        /// </summary>
        [Obsolete("FriendlyName is deprecated. Use UniqueName for ONNX names or Key.ToString() for stable identifiers.")]
        public string? FriendlyName => this.uniqueName;

        public override string ToString()
        {
            return (this.uniqueName ?? "") + ": " + this.GetType().Name;
        }

        /// <summary>
        /// Reinterprets this node to the generic node view of element type <typeparamref name="V"/>.
        /// During the migration the concrete nodes are still the generic <see cref="ImmutableVariable{T}"/>
        /// subclasses, so this is a checked reference reinterpret; the element <see cref="DType"/> must
        /// already match.
        /// </summary>
        public ImmutableVariable<V> As<V>() where V : IVarType
        {
            if (this is ImmutableVariable<V> v)
                return v;

            throw new InvalidTensorOperationException(ErrorCodes.CR006, "As<V>", $"from {this.Type} to {typeof(V).Name}",
                $"Cannot reinterpret {this.GetType().Name} as ImmutableVariable<{typeof(V).Name}> - element types are not compatible");
        }
    }

    public abstract class ImmutableVariable<T> : Variable, IValue<T> where T : IVarType
    {
        public ImmutableVariable(DType type, Node owningNode, Function? moduleFn, string? name)
            : base(type, owningNode, moduleFn, name)
        {
        }

        public Tensor<T> Tensor() => (ImmutableTensor<T>)this;
        public Vector<T> Vec() => this.Tensor().Vec();
        public Scalar<T> Scalar() => this.Tensor().Scalar();
        public TensorSequence<T> Sequence() => (ImmutableTensorSequence<T>)this;

        public override string ToString()
        {
            return base.ToString() + "[" + (this.Tensor().Rank ?? -1) + "]";
        }
    }
}

#pragma warning restore CS8981
