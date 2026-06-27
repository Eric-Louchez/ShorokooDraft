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
