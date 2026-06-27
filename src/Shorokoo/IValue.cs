
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
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using static Shorokoo.Core.Nodes.Ops;
using static Shorokoo.Core.Nodes.AutoDiff.Ops;
using static RandN.Distributions.Uniform;
using System.Diagnostics.CodeAnalysis;

namespace Shorokoo
{
    public interface IModuleParam
    {
    }

    public interface IValue : IModuleParam
    {
        public Node OwningNode { get; }

        public Node ParentNode => this.OwningNode;
        public DType Type { get; }
        public DType DType => this.Type;
        public Tensor<V> As<V>() where V : IVarType;
        public Function? ModuleFn { get; }

        /// <summary>
        /// The backing graph-side <see cref="Variable"/> node this handle wraps, or <c>null</c> for a
        /// defaulted/absent handle. Lets framework machinery recover the node from a boxed handle whose
        /// concrete type has been erased (see <see cref="Shorokoo.Core.VariableHandle.Normalize"/>).
        /// </summary>
        Variable? Immutable { get; }
        
        /// <summary>
        /// The unique name for this tensor. Defaults to Key.ToString() but can be set to human-readable
        /// names like "N1_T0" by processors during construction. Used for ONNX serialization.
        /// </summary>
        string UniqueName { get; }
        
        /// <summary>
        /// Obsolete: Use UniqueName instead. DefaultName now redirects to UniqueName for backwards compatibility.
        /// </summary>
        [Obsolete("Use UniqueName instead. DefaultName is deprecated and will be removed in a future version.")]
        string DefaultName => UniqueName;
        
        /// <summary>
        /// Deprecated: FriendlyName is no longer used. Use UniqueName for ONNX names or Key for stable identifiers.
        /// </summary>
        [Obsolete("FriendlyName is deprecated. Use UniqueName for ONNX names or Key.ToString() for stable identifiers.")]
        string? FriendlyName { get; }
        
        bool IsValid { get; set; }

        /// <summary>
        /// A globally unique identifier for this tensor, composed of the parent node's key and the output index.
        /// </summary>
        TensorKey Key { get; }

        bool IsConnectingTensor => OwningNode.IsOpenNode && OwningNode.ConnectingTensor == this;

        InputType? InputType
        {
            get
            {
                if (!this.OwningNode.IsModelInput)
                    return null;

                if (this.OwningNode.OpCode == InternalOpCodes.GENERIC_TYPE_INPUT)
                    return Shorokoo.Core.Nodes.NodeDefinitions.InputType.GenericType;

                var inputType = this.OwningNode.Attributes.GetEnumVal<Shorokoo.Core.Nodes.NodeDefinitions.InputType>(OnnxOpAttributeNames.ShrkAttrInputType);
                Debug.Assert(inputType != Shorokoo.Core.Nodes.NodeDefinitions.InputType.ModelInput); // Not really supported yet.

                return inputType ?? Shorokoo.Core.Nodes.NodeDefinitions.InputType.ReadyInput;
            }
        }

        /// <summary>
        /// The <c>[Hyper(defaultValue)]</c> default recorded on this model-input node, or null when
        /// the input is not a defaulted hyperparameter. Lets serializers re-emit the default (e.g. the
        /// C# emitter writes <c>[Hyper(defaultValue)]</c>). The attribute is optional, so a node that
        /// never carried it (any non-defaulted input) reads back as null rather than throwing.
        /// </summary>
        float? HyperDefaultValue
            => this.OwningNode.IsModelInput
                && this.OwningNode.Attributes.GetAttributeVals().TryGetValue(OnnxOpAttributeNames.ShrkAttrDefaultValue, out var dv)
                ? (float?)dv
                : null;

        TensorDim[]? TensorDims 
        {
            get
            {
                var tensorData = this.OwningNode.GetTensorData();
                if (tensorData is not null)
                    return tensorData.Shape.Dims.Select(x => new TensorDim(x)).ToArray();

                var rank = this.Rank();
                if (rank is null)
                    return null;

                return Enumerable.Range(1, rank.AssertNotNull()).Select(x => new TensorDim()).ToArray();
            }
        }
    }

    /// <summary>
    /// Interface for TensorStruct instances. TensorStruct is a mechanism for grouping multiple IValues together into a single composite IValue.
    /// Parallel to ITensor, ITensorSequence, and IOptionalTensor.
    /// </summary>
    public interface ITensorStruct : IValue
    {
        /// <summary>
        /// Gets the definition describing the structure of this TensorStruct (field names, types, order).
        /// </summary>
        TensorStructDef Definition { get; }

        /// <summary>
        /// Gets a field from this TensorStruct by name.
        /// </summary>
        /// <param name="name">The name of the field to retrieve</param>
        /// <returns>The graph node (<see cref="Variable"/>) for the specified field; it
        /// implicitly converts to the typed handle the caller expects.</returns>
        Variable GetField(string name);
    }

    public static class IValueExtensions
    {
        // Element-type reinterprets — produce the typed tensor handle of element type T over the same
        // graph node (the node itself is non-generic; the runtime dtype is unchanged).
        public static Tensor<T> As<T>(this IValue var) where T : IVarType => Tensor<T>.Reinterpret(var.ToVariable());
        public static Tensor<uint4> uint4(this IValue var) => Tensor<uint4>.Reinterpret(var.ToVariable());
        public static Tensor<uint8> uint8(this IValue var) => Tensor<uint8>.Reinterpret(var.ToVariable());
        public static Tensor<uint16> uint16(this IValue var) => Tensor<uint16>.Reinterpret(var.ToVariable());
        public static Tensor<uint32> uint32(this IValue var) => Tensor<uint32>.Reinterpret(var.ToVariable());
        public static Tensor<uint64> uint64(this IValue var) => Tensor<uint64>.Reinterpret(var.ToVariable());
        public static Tensor<int4> int4(this IValue var) => Tensor<int4>.Reinterpret(var.ToVariable());
        public static Tensor<int8> int8(this IValue var) => Tensor<int8>.Reinterpret(var.ToVariable());
        public static Tensor<int16> int16(this IValue var) => Tensor<int16>.Reinterpret(var.ToVariable());
        public static Tensor<int32> int32(this IValue var) => Tensor<int32>.Reinterpret(var.ToVariable());
        public static Tensor<int64> int64(this IValue var) => Tensor<int64>.Reinterpret(var.ToVariable());
        public static Tensor<float16> float16(this IValue var) => Tensor<float16>.Reinterpret(var.ToVariable());
        public static Tensor<bfloat16> bfloat16(this IValue var) => Tensor<bfloat16>.Reinterpret(var.ToVariable());
        public static Tensor<float32> float32(this IValue var) => Tensor<float32>.Reinterpret(var.ToVariable());
        public static Tensor<float64> float64(this IValue var) => Tensor<float64>.Reinterpret(var.ToVariable());

        public static DataStructure Structure(this IValue var)
            => var is Variable nodeVar ? nodeVar.Kind :
               var is ITensorStruct ? DataStructure.TensorStruct :
               var is ITensor ? DataStructure.Tensor :
               var is IOptionalTensor ? DataStructure.Optional :
               DataStructure.Sequence;

        public static int? Rank(this IValue var)
            => var is ITensor tensor ? tensor.Rank : null;

        public static bool IsModelInput(this IValue var) => var.OwningNode.IsModelInput;

        // Unwrap a module parameter to its graph-side Variable node. Handles (IValue) carry their
        // backing node via IValue.Immutable; models/modules expose it through their *Variable handle.
        internal static Variable ToVariable(this IModuleParam param) =>
                        param switch
                        {
                            IModel model => (Variable)model.ModelVariable,
                            IModule module => (Variable)module.ModuleVariable,
                            IValue handle => handle.Immutable
                                 ?? throw new InvalidTensorOperationException(ErrorCodes.CR001, "ToVariable", param.GetType().Name,
                                        "Invalid IModuleParam type for variable conversion"),
                            _ => throw new InvalidTensorOperationException(ErrorCodes.CR001, "ToVariable", param?.GetType()?.Name ?? "null",
                                        "Invalid IModuleParam type for variable conversion"),
                        };
    }
}
