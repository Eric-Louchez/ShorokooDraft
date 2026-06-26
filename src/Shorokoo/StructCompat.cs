using Shorokoo.Core;
using Shorokoo.Core.Nodes.NodeDefinitions;

namespace Shorokoo
{
    // Compatibility shims that restore behaviour the value-struct conversion would otherwise
    // lose to the absence of class inheritance, so user/module code keeps compiling unchanged:
    //   * Scalar/Vector are no longer subclasses of Tensor, so generic inference that used to flow
    //     through the base type (e.g. Pow<T1>(Tensor<T1>) called with a Scalar<T1>) needs explicit
    //     overloads taking the concrete struct.
    //   * `struct Scalar<T>` cannot declare a member named `Scalar` (CS0542), so the inherited
    //     `.Scalar()` reinterpret becomes an extension.
    public partial struct Tensor<T> where T : IVarType
    {
        /// <summary>Element-wise power with a scalar exponent (overload so T1 infers from a <see cref="Scalar{T1}"/>).</summary>
        public Tensor<T> Pow<T1>(Scalar<T1> power) where T1 : IVarType => this.Pow<T1>((Tensor<T1>)power);
        /// <summary>Element-wise power with a vector exponent (overload so T1 infers from a <see cref="Vector{T1}"/>).</summary>
        public Tensor<T> Pow<T1>(Vector<T1> power) where T1 : IVarType => this.Pow<T1>((Tensor<T1>)power);
    }

    public partial struct Scalar<T> where T : IVarType
    {
        /// <summary>Power with a tensor exponent — the base broadcasts (former <c>Scalar : Tensor</c> inheritance).</summary>
        public Tensor<T> Pow<T1>(Tensor<T1> power) where T1 : IVarType => ((Tensor<T>)this).Pow<T1>(power);
        /// <summary>Power with a vector exponent — the base broadcasts (overload so T1 infers from a <see cref="Vector{T1}"/>).</summary>
        public Tensor<T> Pow<T1>(Vector<T1> power) where T1 : IVarType => ((Tensor<T>)this).Pow<T1>((Tensor<T1>)power);
    }

    public static class ValueStructCompatExtensions
    {
        /// <summary>Identity reinterpret of an already-rank-0 scalar (the inherited <c>Scalar()</c> on the
        /// former <c>Scalar : Tensor</c>; the struct cannot declare a same-named member).</summary>
        public static Scalar<T> Scalar<T>(this Scalar<T> self) where T : IVarType => self;

        /// <summary>Element-wise select driven by a scalar condition (the <c>Where</c> extension requires a
        /// <see cref="Tensor{bit}"/> receiver; a <see cref="Scalar{bit}"/> reaches it via this overload).</summary>
        public static Tensor<V> Where<V>(this Scalar<bit> cond, Tensor<V> whenTrue, Tensor<V> whenFalse)
            where V : IVarType
            => ((Tensor<bit>)cond).Where(whenTrue, whenFalse);
    }
}
