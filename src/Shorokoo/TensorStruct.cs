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
    /// Immutable (class) graph node for a TensorStruct — a composite that groups multiple
    /// <see cref="IVariable"/>s into one. This is the value the graph stores; it holds the field
    /// data and definition and satisfies the <see cref="ITensorStruct"/> contract. The user-facing
    /// API lives on the value-type handle <see cref="TensorStruct{T}"/>.
    /// </summary>
    /// <typeparam name="T">The IStruct type that defines the struct fields. Use DTypeStruct for dynamic struct definitions.</typeparam>
    public class ImmutableTensorStruct<T> : Variable<T>, ITensorStruct where T : IStruct
    {
        private readonly ImmutableDictionary<string, IVariable> _fields;
        private readonly TensorStructDef _definition;

        internal ImmutableTensorStruct(DType dtype, Node owningNode, Function? moduleFn, string? name,
            TensorStructDef definition, ImmutableDictionary<string, IVariable>? fields = null)
            : base(dtype, owningNode, moduleFn, name)
        {
            _definition = definition ?? throw new ArgumentNullException(nameof(definition));
            _fields = fields ?? ImmutableDictionary<string, IVariable>.Empty;
        }

        // ITensorStruct contract (the minimal graph-node surface).
        TensorStructDef ITensorStruct.Definition => _definition;
        IVariable ITensorStruct.GetField(string name) => Field(name);

        // Internal accessors used by the value-struct handle to build the public surface.
        internal TensorStructDef Def => _definition;
        internal ImmutableDictionary<string, IVariable> Fields => _fields;

        internal IVariable Field(string name)
        {
            if (_fields.TryGetValue(name, out var field))
                return field;

            throw new KeyNotFoundException($"Field '{name}' not found in TensorStruct. Available fields: {string.Join(", ", _fields.Keys)}");
        }

        internal ImmutableTensorStruct<T> WithFields(ImmutableDictionary<string, IVariable> newFields)
            => new ImmutableTensorStruct<T>(this.Type, this.OwningNode, this.ModuleFn, this.UniqueName, _definition, newFields);

        public override string ToString()
        {
            var typeName = _definition.TypeName ?? "DTypeStruct";
            return $"TensorStruct<{typeName}>[{_fields.Count} fields]";
        }
    }

    /// <summary>
    /// Value-type handle for a TensorStruct. The original <c>TensorStruct&lt;T&gt;</c> name now denotes
    /// this <see langword="struct"/>; the reference type was renamed <see cref="ImmutableTensorStruct{T}"/>.
    /// This struct carries the full user-facing surface. It holds the immutable directly in a field
    /// (value-copy semantics for the Module DSL). This pass only makes mutation possible — behaviour
    /// is unchanged (de-facto immutable).
    /// <para>
    /// A defaulted handle (<c>default</c>, <c>inner == null</c>) has no field layout to materialise, so
    /// accessing it throws — a TensorStruct must be produced by a graph op (e.g. <c>Globals.TensorStruct</c>).
    /// </para>
    /// </summary>
    public struct TensorStruct<T> : ITensorStruct where T : IStruct
    {
        private ImmutableTensorStruct<T>? inner;

        /// <summary>The wrapped immutable. A defaulted handle has no recoverable field layout, so this throws.</summary>
        internal readonly ImmutableTensorStruct<T> Imm
            => inner ?? throw new InvalidOperationException(
                "default(TensorStruct<T>) has no field layout; create one via a graph op (e.g. Globals.TensorStruct<T>(...)).");

        public static implicit operator TensorStruct<T>(ImmutableTensorStruct<T> imm)
            => new TensorStruct<T> { inner = imm };
        public static implicit operator ImmutableTensorStruct<T>(TensorStruct<T> handle)
            => handle.Imm;

        // ── User-facing API (the struct surface lives here, not on the immutable) ──
        public TensorStructDef Definition => Imm.Def;

        public IVariable GetField(string name) => Imm.Field(name);

        public TField GetField<TField>(string name) where TField : IVariable
        {
            var field = Imm.Field(name);
            if (field is TField typedField)
                return typedField;

            throw new InvalidCastException($"Field '{name}' is of type {field.GetType().Name}, not {typeof(TField).Name}");
        }

        public bool TryGetField(string name, out IVariable? field) => Imm.Fields.TryGetValue(name, out field);

        public IEnumerable<string> FieldNames => Imm.Fields.Keys;

        public IEnumerable<KeyValuePair<string, IVariable>> AllFields => Imm.Fields;

        internal TensorStruct<T> WithFields(ImmutableDictionary<string, IVariable> newFields) => Imm.WithFields(newFields);

        public override readonly string ToString() => Imm.ToString();

        // ITensorStruct explicit members.
        TensorStructDef ITensorStruct.Definition => Imm.Def;
        IVariable ITensorStruct.GetField(string name) => Imm.Field(name);

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
