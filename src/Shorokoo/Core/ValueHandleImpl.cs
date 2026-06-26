using Shorokoo.Core;

namespace Shorokoo
{
    // Expose the backing Immutable* graph value so framework machinery can normalise a boxed
    // struct handle back to the immutable it wraps (see Shorokoo.Core.VariableHandle). A defaulted
    // handle (inner == null) is an absent value, so it normalises to null rather than throwing.
    public partial struct Tensor<T> : IValueHandle where T : IVarType
    {
        IVariable? IValueHandle.Immutable => inner;
    }

    public partial struct Vector<T> : IValueHandle where T : IVarType
    {
        IVariable? IValueHandle.Immutable => inner;
    }

    public partial struct Scalar<T> : IValueHandle where T : IVarType
    {
        IVariable? IValueHandle.Immutable => inner;
    }
}
