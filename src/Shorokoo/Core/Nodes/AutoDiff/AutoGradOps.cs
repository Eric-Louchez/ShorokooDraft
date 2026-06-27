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
            => InternalOp.AutoGrad([input.ToVariable()], loss)[0]!.Cast<A>();

        public static (A?, B?) AutoGrad<A, B, T>(A? a, B? b, Scalar<T> loss)
            where A : IValue
            where B : IValue
            where T : FloatLike
        {
            var retval = InternalOp.AutoGrad([Shorokoo.Core.VariableHandle.Normalize(a), Shorokoo.Core.VariableHandle.Normalize(b)], loss);
            return (retval[0] is null ? default! : retval[0].Cast<A>(),
                    retval[1] is null ? default! : retval[1].Cast<B>());
        }

        public static Variable?[] AutoGrad<T>(Variable?[] inputs, Scalar<T> loss) where T : FloatLike
            => InternalOp.AutoGrad(inputs, loss);

        public static Variable?[] AutoGrad<T>(IValue?[] inputs, Scalar<T> loss) where T : FloatLike
            => InternalOp.AutoGrad([.. System.Linq.Enumerable.Select(inputs, v => v?.Immutable)], loss);
    }
}
