using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
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

namespace Shorokoo
{

    /// <summary>
    /// Value-type handle for a TensorStruct. The original <c>TensorStruct&lt;T&gt;</c> name now denotes
    /// this <see langword="struct"/>; the reference type was renamed <see cref="Variable"/>.
    /// This struct carries the full user-facing surface. It holds the immutable directly in a field
    /// (value-copy semantics for the Module DSL). This pass only makes mutation possible — behaviour
    /// is unchanged (de-facto immutable).
    /// <para>
    /// A defaulted handle (<c>default</c>, <c>inner == null</c>) has no field layout to materialise, so
    /// accessing it throws — a TensorStruct must be produced by a graph op (e.g. <c>Globals.TensorStruct</c>).
    /// </para>
    /// </summary>
    public struct TensorStruct<T> : ITensorStruct, Shorokoo.Core.IValueHandle where T : IStruct
    {
        private Variable? inner;

        Variable Shorokoo.Core.IValueHandle.Immutable => Imm;

        /// <summary>The wrapped immutable. A defaulted handle has no recoverable field layout, so this throws.</summary>
        internal readonly Variable Imm
            => inner ?? throw new InvalidOperationException(
                "default(TensorStruct<T>) has no field layout; create one via a graph op (e.g. Globals.TensorStruct<T>(...)).");

        public static implicit operator TensorStruct<T>(Variable imm)
            => new TensorStruct<T> { inner = imm };
        public static implicit operator Variable(TensorStruct<T> handle)
            => handle.Imm;

        // ── User-facing API (the struct surface lives here, not on the immutable) ──
        public TensorStructDef Definition => Imm.Def;

        public Variable GetField(string name) => Imm.Field(name);

        public TField GetField<TField>(string name) where TField : IValue
            => VariableHandle.Cast<TField>(Imm.Field(name));

        public bool TryGetField(string name, out Variable? field) => Imm.Fields.TryGetValue(name, out field);

        public IEnumerable<string> FieldNames => Imm.Fields.Keys;

        public IEnumerable<KeyValuePair<string, Variable>> AllFields => Imm.Fields;

        internal TensorStruct<T> WithFields(ImmutableDictionary<string, Variable> newFields) => Imm.WithFields(newFields);

        public override readonly string ToString() => Imm.ToString();

        // ITensorStruct explicit members.
        TensorStructDef ITensorStruct.Definition => Imm.Def;
        Variable ITensorStruct.GetField(string name) => Imm.Field(name);

        // IValue surface — forward to the wrapped immutable.
        public Node OwningNode => Imm.OwningNode;
        public DType Type => Imm.Type;
        public Function? ModuleFn => Imm.ModuleFn;
        public TensorKey Key => Imm.Key;
        public string UniqueName => Imm.UniqueName;
        public bool IsValid { get => Imm.IsValid; set => Imm.IsValid = value; }
        public Tensor<V> As<V>() where V : IVarType => ((IValue)Imm).As<V>();

#pragma warning disable CS0618 // forwarding the obsolete member is intentional
        string? IValue.FriendlyName => ((IValue)Imm).FriendlyName;
#pragma warning restore CS0618
    }
}
