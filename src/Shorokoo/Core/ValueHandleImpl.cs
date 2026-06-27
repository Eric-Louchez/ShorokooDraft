using Shorokoo.Core;

namespace Shorokoo
{
    // Expose the backing graph Variable so framework machinery can normalise a boxed struct handle
    // back to the node it wraps (see Shorokoo.Core.VariableHandle). A defaulted handle (inner == null)
    // is an absent value, so it normalises to null rather than throwing.
    public partial struct Tensor<T> where T : IVarType
    {
        Variable? IValue.Immutable => inner;
    }

    public partial struct Vector<T> where T : IVarType
    {
        Variable? IValue.Immutable => inner;
    }

    public partial struct Scalar<T> where T : IVarType
    {
        Variable? IValue.Immutable => inner;
    }
}
