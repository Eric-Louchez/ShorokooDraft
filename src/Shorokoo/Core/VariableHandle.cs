using System;
using System.Collections.Concurrent;
using System.Reflection;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo.Core
{
    /// <summary>
    /// Converts an <see cref="Variable"/> graph value (always one of the <c>Immutable*</c> classes)
    /// into a requested handle type. When the target is one of the value-type handles
    /// (<see cref="Tensor{T}"/>, <see cref="Vector{T}"/>, <see cref="Scalar{T}"/>, …) a plain
    /// <c>(A)value</c> cast would unbox an interface reference to a struct and throw; this finds and
    /// invokes the immutable→struct implicit conversion instead. Interface / immutable-class targets
    /// still go through the fast direct-cast path.
    /// </summary>
    public static class VariableHandle
    {
        private static readonly ConcurrentDictionary<Type, MethodInfo[]> wrappers = new();

        /// <summary>The backing <c>Immutable*</c> graph value: a struct handle unwrapped, an immutable
        /// as-is, or null for a defaulted/absent struct handle.</summary>
        public static Variable? Normalize(object? value)            => value is IValue h ? h.Immutable : value as Variable;

        /// <summary>
        /// Validate and adapt a graph <paramref name="node"/> for wrapping into a typed value handle —
        /// the invariant the implicit <c>Variable</c>→handle operators must uphold:
        /// <list type="bullet">
        /// <item>the node's structural <see cref="DataStructure"/> equals <paramref name="kind"/>
        ///   (structure must always match);</item>
        /// <item>the node's runtime element <see cref="DType"/> equals <paramref name="expected"/> — an
        ///   implicit dtype change is rejected (use <c>Cast&lt;T&gt;()</c> to convert or
        ///   <c>As&lt;T&gt;()</c> to reinterpret). Skipped when either side is a generic-placeholder,
        ///   unmapped, or invalid dtype (e.g. inside a generic module before specialization);</item>
        /// <item>for the fixed-rank tensor handles (<see cref="Scalar{T}"/> = rank 0,
        ///   <see cref="Vector{T}"/> = rank 1) a matching rank passes through, an <b>unknown</b> rank is
        ///   adapted with an <c>Identity</c> rank-conversion node, and a known mismatching rank is an
        ///   error.</item>
        /// </list>
        /// </summary>
        internal static Variable ForHandle(Variable node, DType? expected, DataStructure kind, int? fixedRank)
        {
            if (node.Kind != kind)
                throw new InvalidTensorOperationException(ErrorCodes.CR011, "Variable→handle conversion",
                    node.Kind.ToString(),
                    $"cannot wrap a {node.Kind} graph value in a {kind} handle — structure must match");

            if (expected is not null && expected.IsValid && node.Type.IsValid
                && !expected.IsGenericType && !node.Type.IsGenericType
                && !expected.IsGenericTypeReference && !node.Type.IsGenericTypeReference
                && !node.Type.Equals(expected))
                throw new InvalidTensorOperationException(ErrorCodes.CR012, "Variable→handle conversion",
                    $"{node.Type} as {expected}",
                    $"element-type mismatch — wrapping a {node.Type} graph value as a {expected} handle would silently " +
                    $"reinterpret it; use Cast<{expected}>() to convert the dtype or As<{expected}>() to reinterpret");

            if (fixedRank is int r)
            {
                if (node.Rank == r) return node;
                if (node.Rank is null) return OnnxOp.Identity(node, r);
                throw new InvalidTensorOperationException(ErrorCodes.CR013, "Variable→handle conversion",
                    $"rank {node.Rank} as rank {r}",
                    $"cannot wrap a rank-{node.Rank} tensor as a rank-{r} handle");
            }
            return node;
        }

        /// <summary>
        /// Wrap a graph value into a value-struct parameter type for reflective invocation
        /// (<c>MethodInfo.Invoke</c> does not apply user-defined conversions). Non-struct or
        /// already-matching parameters pass through unchanged.
        /// </summary>
        public static object? WrapForParam(object? value, Type paramType)
        {
            if (value is null)
                return value;

            var imm = Normalize(value);
            if (imm is null)
                // A defaulted/absent struct handle: hand back the value as-is (it already matches a
                // struct-typed parameter; reflection boxes it).
                return value;

            // Parameter wants an immutable / interface — hand it the unwrapped graph value.
            if (paramType.IsInstanceOfType(imm))
                return imm;

            // Already the requested struct handle.
            if (paramType.IsInstanceOfType(value))
                return value;

            // A struct handle parameter may be declared nullable (`Tensor<T>?`), which on a value
            // type is Nullable<Tensor<T>>; convert to the underlying handle (reflection accepts a
            // boxed T for a Nullable<T> parameter).
            var target = Nullable.GetUnderlyingType(paramType) ?? paramType;
            if (target.IsValueType && typeof(IValue).IsAssignableFrom(target))
            {
                var conv = MatchingConverter(target, imm.GetType());
                if (conv != null)
                    return conv.Invoke(null, [imm]);
            }

            return value;
        }

        /// <summary>
        /// Reflection core of <see cref="Variable.Cast{A}"/>: wrap <paramref name="node"/> into the value
        /// handle <paramref name="handleType"/> by invoking that handle's <c>op_Implicit(Variable)</c>
        /// (which validates structure/dtype/rank). Constraint-free so it also serves call sites whose
        /// target type is statically unconstrained (e.g. <c>ModuleHelper.Reformat&lt;T&gt;</c>). When no
        /// converter exists the node is returned boxed, so the caller's cast raises a clear error.
        /// </summary>
        internal static object WrapAsHandle(Variable node, Type handleType)
        {
            var conv = MatchingConverter(handleType, node.GetType());
            return conv is not null ? conv.Invoke(null, [node])! : node;
        }

        internal static MethodInfo? MatchingConverter(Type handleType, Type valueType)
        {
            var candidates = wrappers.GetOrAdd(handleType, FindImplicitWrappers);
            foreach (var m in candidates)
            {
                if (m.GetParameters()[0].ParameterType.IsAssignableFrom(valueType))
                    return m;
            }
            return null;
        }

        // Implicit conversion operators that PRODUCE the handle type from a reference (Immutable*) value.
        private static MethodInfo[] FindImplicitWrappers(Type handleType)
        {
            var result = new System.Collections.Generic.List<MethodInfo>();
            foreach (var m in handleType.GetMethods(BindingFlags.Public | BindingFlags.Static))
            {
                if (m.Name != "op_Implicit" || m.ReturnType != handleType)
                    continue;
                var p = m.GetParameters();
                if (p.Length != 1)
                    continue;
                var pt = p[0].ParameterType;
                if (!pt.IsValueType && typeof(Variable).IsAssignableFrom(pt))
                    result.Add(m);
            }
            return result.ToArray();
        }
    }
}
