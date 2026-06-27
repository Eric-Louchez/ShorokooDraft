
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
using System.Collections.Immutable;

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
    /// During this migration <see cref="Variable"/> still implements <see cref="IValue"/> — the
    /// graph's internal currency — so the collapse stays green without flipping every op signature.
    /// The final, separable step removes <c>IValue</c> from the node so that <c>IValue</c> becomes
    /// purely the user-facing handle interface (implemented by the value-struct handles such as
    /// <see cref="Tensor{T}"/>) and a graph node is no longer a handle.
    /// </para>
    /// </summary>
    public class Variable : IValue, ITensor, IVector, IScalar, ITensorSequence, IOptionalTensor, ITensorStruct
    {
        public Node OwningNode { get; private set; }

        public DType Type { get; private set; }

        public Function? ModuleFn { get; private set; }

        public bool IsValid { get; set; } = true;

        /// <summary>The structural kind of this graph value (tensor / optional / sequence / struct).
        /// Tensor/vector/scalar all share <see cref="DataStructure.Tensor"/> and are distinguished by
        /// <see cref="Rank"/>.</summary>
        public DataStructure Kind { get; }

        /// <summary>Statically known rank (number of dimensions), or null when not known at
        /// graph-construction time. Only meaningful for <see cref="DataStructure.Tensor"/> values.</summary>
        public int? Rank { get; }

        /// <summary>
        /// A globally unique identifier for this tensor, composed of the parent node's key and the output index.
        /// Set by the Node constructor after creating outputs.
        /// </summary>
        public TensorKey Key { get; private set; }

        private string? uniqueName;
        private readonly Func<Vector<int64>>? shapeInferer;
        private Vector<int64>? infShapeTensor;
        private readonly TensorStructDef? structDef;
        private readonly ImmutableDictionary<string, IValue> fields;

        internal Variable(DType type, Node owningNode, Function? moduleFn, string? name,
            DataStructure kind = DataStructure.Tensor, int? rank = null,
            Func<Vector<int64>>? shapeFn = null,
            TensorStructDef? structDef = null,
            ImmutableDictionary<string, IValue>? fields = null)
        {
            this.OwningNode = owningNode;
            this.Type = type;
            this.uniqueName = name;
            this.ModuleFn = moduleFn;
            this.Kind = kind;
            this.Rank = rank;
            this.shapeInferer = shapeFn;
            this.structDef = structDef;
            this.fields = fields ?? ImmutableDictionary<string, IValue>.Empty;
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

        /// <summary>
        /// Reinterprets this node as the typed tensor handle of element type <typeparamref name="V"/>.
        /// The node is non-generic; the runtime element <see cref="DType"/> is unchanged (this is a
        /// static-type reinterpret, not a dtype conversion — use <c>Cast</c> to convert).
        /// </summary>
        public Tensor<V> As<V>() where V : IVarType => this;

        // ── ITensor surface (meaningful for DataStructure.Tensor values) ──
        public virtual Vector<int64> DShape => (Variable)OnnxOp.Shape(this, null, null);

        public virtual Vector<int64>? InfShape => this.Kind == DataStructure.Tensor && this.Rank == 0
            ? Vector<int64>.Empty
            : (this.infShapeTensor ??= this.shapeInferer?.Invoke());

        public Vector<int64> TShape => this.DShape;
        public Scalar<int64> TRank => TShape.TShape[0].T;

        Vector<V> ITensor.Vec<V>() => this.Cast<V>().Vec();
        Scalar<V> ITensor.Scalar<V>() => this.Cast<V>().Scalar();

        /// <summary>Reinterprets this tensor as a rank-1 vector, inserting an Identity node when needed.</summary>
        public IVector Vec() => this.Rank == 1 ? this : (Variable)OnnxOp.Identity(this, rank: 1);

        /// <summary>Reinterprets this tensor as a rank-0 scalar, inserting an Identity node when needed.</summary>
        public IScalar Scalar() => this.Rank == 0 ? this : (Variable)OnnxOp.Identity(this, rank: 0);

        /// <summary>Casts the element type to <typeparamref name="V"/>; returns this tensor unchanged when the types already match.</summary>
        public Tensor<V> Cast<V>(bool saturate = true) where V : IVarType
            => OnnxUtils.GetDType<V>() == this.Type ?
                (Tensor<V>)this :
                (Variable)OnnxOp.Cast(this, saturate ? null : saturate, OnnxUtils.GetDType<V>());

        // ── ITensorSequence surface (meaningful for DataStructure.Sequence values) ──
        Scalar<int64> ITensorSequence.Count => (Variable)OnnxOp.SequenceLength(this);
        ITensor ITensorSequence.Concat(long axis, bool newAxis) => (ITensor)OnnxOp.ConcatFromSequence(this, axis, newAxis);
        ITensor ITensorSequence.this[Scalar<int64> index] => (ITensor)OnnxOp.SequenceAt(this, index);
        ITensorSequence ITensorSequence.RemoveAt(Scalar<int64> index) => (ITensorSequence)OnnxOp.SequenceErase(this, index);
        ITensorSequence ITensorSequence.InsertAt(ITensor tensor, Scalar<int64> index) => (ITensorSequence)OnnxOp.SequenceInsert(this, tensor, index);

        // ── ITensorStruct surface (meaningful for DataStructure.TensorStruct values) ──
        TensorStructDef ITensorStruct.Definition => this.structDef!;
        IValue ITensorStruct.GetField(string name) => Field(name);
        internal TensorStructDef Def => this.structDef!;
        internal ImmutableDictionary<string, IValue> Fields => this.fields;

        internal IValue Field(string name)
        {
            if (this.fields.TryGetValue(name, out var field))
                return field;
            throw new KeyNotFoundException($"Field '{name}' not found in TensorStruct. Available fields: {string.Join(", ", this.fields.Keys)}");
        }

        internal Variable WithFields(ImmutableDictionary<string, IValue> newFields)
            => new Variable(this.Type, this.OwningNode, this.ModuleFn, this.UniqueName, DataStructure.TensorStruct, structDef: this.structDef, fields: newFields);

        public override string ToString()
            => this.Kind == DataStructure.TensorStruct
                ? $"TensorStruct<{this.structDef?.TypeName ?? "DTypeStruct"}>[{this.fields.Count} fields]"
                : (this.uniqueName ?? "") + ": " + this.GetType().Name
                    + (this.Kind == DataStructure.Tensor ? "[" + (this.Rank ?? -1) + "]" : "/" + this.Kind);
    }
}

#pragma warning restore CS8981
