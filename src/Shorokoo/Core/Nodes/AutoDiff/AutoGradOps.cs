using RandN.Distributions.UnitInterval;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.OnnxNodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Nodes.NodeDefinitions;
using Shorokoo.Modules;
using System;
using System.Collections.Generic;
using System.Text;
using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Training;

namespace Shorokoo.Core.Nodes.AutoDiff
{
    public static partial class Ops
    {
        public static A AutoGrad<A, T>(A input, Scalar<T> loss) where A : IValue where T : FloatLike
            => Shorokoo.Core.VariableHandle.Cast<A>(InternalOp.AutoGrad([input], loss)[0]!);

        public static (A?, B?) AutoGrad<A, B, T>(A? a, B? b, Scalar<T> loss)
            where A : IValue
            where B : IValue
            where T : FloatLike
        {
            var retval = InternalOp.AutoGrad([a, b], loss);
            return (Shorokoo.Core.VariableHandle.Cast<A>(retval[0]), Shorokoo.Core.VariableHandle.Cast<B>(retval[1]));
        }

        public static IValue?[] AutoGrad<T>(IValue?[] inputs, Scalar<T> loss) where T : FloatLike
            => InternalOp.AutoGrad(inputs, loss);
    }
}
