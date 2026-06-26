using System;
using System.Collections.Concurrent;
using System.Reflection;

namespace Shorokoo.Core
{
    /// <summary>
    /// Implemented by every value-type handle (<see cref="Tensor{T}"/>, <see cref="Vector{T}"/>, …)
    /// so the framework can recover the backing <c>Immutable*</c> graph value from a boxed handle.
    /// Graph-level machinery must hold immutables, never struct handles, as <see cref="IVariable"/>.
    /// </summary>
    internal interface IValueHandle
    {
        // The backing Immutable* graph value, or null for a defaulted/absent handle.
        IVariable? Immutable { get; }
    }

    /// <summary>
    /// Converts an <see cref="IVariable"/> graph value (always one of the <c>Immutable*</c> classes)
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
        public static IVariable? Normalize(IVariable? value)
            => value is IValueHandle h ? h.Immutable : value;

        /// <summary>Reinterpret <paramref name="value"/> as the handle type <typeparamref name="A"/>.</summary>
        public static A Cast<A>(IVariable? value)
        {
            if (value is A already)
                return already;

            if (value is null)
                return default!;

            var imm = Normalize(value);
            if (imm is null)
                return default!;
            if (imm is A immA)
                return immA;

            var conv = MatchingConverter(typeof(A), imm.GetType());
            if (conv != null)
                return (A)conv.Invoke(null, [imm])!;

            // No struct wrapper found — fall back to a direct cast so the runtime raises a clear error.
            return (A)imm;
        }

        /// <summary>
        /// Wrap a graph value into a value-struct parameter type for reflective invocation
        /// (<c>MethodInfo.Invoke</c> does not apply user-defined conversions). Non-struct or
        /// already-matching parameters pass through unchanged.
        /// </summary>
        public static object? WrapForParam(IVariable? value, Type paramType)
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
            if (target.IsValueType && typeof(IVariable).IsAssignableFrom(target))
            {
                var conv = MatchingConverter(target, imm.GetType());
                if (conv != null)
                    return conv.Invoke(null, [imm]);
            }

            return value;
        }

        private static MethodInfo? MatchingConverter(Type handleType, Type valueType)
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
                if (!pt.IsValueType && typeof(IVariable).IsAssignableFrom(pt))
                    result.Add(m);
            }
            return result.ToArray();
        }
    }
}
