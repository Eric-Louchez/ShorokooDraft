using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static partial class OnnxOp
{
    public static IValue Dft(IValue x, IValue? length, IValue? axis, bool? inverse, bool? onesided = null)
    {
        // ORT 1.25/1.26 segfaults when the DFT node's axis input is omitted (the
        // empty-string placeholder in the protobuf), even though the ONNX spec
        // says the default is -2. Substitute the spec default explicitly so the
        // emitted graph never carries a null axis input.
        axis ??= Shorokoo.Globals.Scalar(-2L);
        return NodeBuilder.BuildNodeSingleOut(DFT, [x, length, axis], [(AttrInverse, inverse), (AttrOnesided, onesided)]);
    }

    public static IValue DeformConv(IValue x, IValue w, IValue offset, IValue? b, IValue? mask,
        long[]? dilations, long? group, long[]? kernelShape, long? offsetGroup,
        long[]? pads, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(DEFORM_CONV, [x, w, offset, b, mask], [
            (AttrDilations, dilations),
            (AttrGroup, group),
            (AttrKernelShape, kernelShape),
            (AttrOffsetGroup, offsetGroup),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IValue DepthToSpace(IValue input, long? blockSize, DepthColumnRowMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(DEPTH_TO_SPACE, [input], [(AttrBlocksize, blockSize), (AttrMode, mode)]);

    public static IValue DequantizeLinear(IValue x, IValue xScale, IValue? xZeroPoint,
        long? axis, long? blockSize = null)
        => NodeBuilder.BuildNodeSingleOut(DEQUANTIZE_LINEAR, [x, xScale, xZeroPoint], 
            [(AttrAxis, axis), (AttrBlockSize, blockSize)]);

    public static IValue Det(IValue x)
        => NodeBuilder.BuildNodeSingleOut(DET, [x], []);

    public static IValue Div(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(DIV, [left, right], []);

    public static (IValue output, IValue? mask) Dropout(IValue data, IValue? ratio, IValue? training_mode,
        long? seed = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(DROPOUT, [data, ratio, training_mode], [(AttrSeed, seed)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null);
    }

    // public static IValue Dropout(IValue data, IValue? ratio, IValue? training_mode,
    //     long? seed = null)
    // {
    //     var retval = NodeBuilder.BuildNodeMultiOut(DROPOUT, [data, ratio, training_mode], [(AttrSeed, seed)], outputNames: [null, ""]);
    //     return (retval[0], retval.Length > 1 ? retval[1] : null);
    // }

    public static (IValue y, IValue yScale, IValue yZeroPoint) DynamicQuantizeLinear(IValue x)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(DYNAMIC_QUANTIZE_LINEAR, [x], []);
        return (retval[0], retval[1], retval[2]);
    }

    public static IValue Einsum(IValue[] inputs, string? equation = null)
        => NodeBuilder.BuildNodeSingleOut(EINSUM, inputs, [(AttrEquation, equation)]);

    public static IValue Elu(IValue x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(ELU, [x], [(AttrAlpha, alpha)]);

    public static IValue Equal(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(EQUAL, [left, right], []);

    public static IValue Erf(IValue x)
        => NodeBuilder.BuildNodeSingleOut(ERF, [x], []);

    public static IValue Exp(IValue x)
        => NodeBuilder.BuildNodeSingleOut(EXP, [x], []);

    public static IValue Expand(IValue input, IValue shape)
        => NodeBuilder.BuildNodeSingleOut(EXPAND, [input, shape], []);

    public static IValue EyeLike(IValue input, DType? dtype = null, long? k = 0)
        => NodeBuilder.BuildNodeSingleOut(EYE_LIKE, [input], [(AttrDtype, dtype), (AttrK, k)]);

    public static IValue Flatten(IValue input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(FLATTEN, [input], [(AttrAxis, axis)]);

    public static IValue Floor(IValue x)
        => NodeBuilder.BuildNodeSingleOut(FLOOR, [x], []);

    public static (IValue y, IValue yH) Gru(IValue x, IValue w, IValue r, 
        IValue? b, IValue? sequenceLens, IValue? initialH,
        float[]? activationAlpha, float[]? activationBeta, string[]? activations,
        float? clip, GRUDirection? direction, long? hiddenSize,
        bool? layout, bool? linearBeforeReset = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(GRU, [x, w, r, b, sequenceLens, initialH], [
            (AttrActivationAlpha, activationAlpha),
            (AttrActivationBeta, activationBeta),
            (AttrActivations, activations),
            (AttrClip, clip),
            (AttrDirection, direction),
            (AttrHiddenSize, hiddenSize),
            (AttrLayout, layout),
            (AttrLinearBeforeReset, linearBeforeReset)]);
        return (retval[0], retval[1]);
    }

    public static IValue Gather(IValue data, IValue indices, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER, [data, indices], [(AttrAxis, axis)]);

    public static IValue GatherElements(IValue data, IValue indices, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER_ELEMENTS, [data, indices], [(AttrAxis, axis)]);

    public static IValue GatherND(IValue data, IValue indices, long? batchDims = null)
        => NodeBuilder.BuildNodeSingleOut(GATHER_ND, [data, indices], [(AttrBatchDims, batchDims)]);

    public static IValue Gelu(IValue x, GeluApproximate? approximate = null)
        => NodeBuilder.BuildNodeSingleOut(GELU, [x], [(AttrApproximate, approximate)]);

    public static IValue GlobalAveragePool(IValue x)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_AVERAGE_POOL, [x], []);

    public static IValue GlobalLpPool(IValue x, long? p = null)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_LP_POOL, [x], [(AttrP, p)]);

    public static IValue GlobalMaxPool(IValue x)
        => NodeBuilder.BuildNodeSingleOut(GLOBAL_MAX_POOL, [x], []);

    public static IValue Greater(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(GREATER, [left, right], []);

    public static IValue GreaterOrEqual(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(GREATER_OR_EQUAL, [left, right], []);

    public static IValue GridSample(IValue x, IValue grid, bool? alignCorners,
        GridSampleMode? mode, GridSamplePaddingMode? paddingMode = null)
        => NodeBuilder.BuildNodeSingleOut(GRID_SAMPLE, [x, grid], [
            (AttrAlignCorners, alignCorners),
            (AttrMode, mode),
            (AttrPaddingMode, paddingMode)]);

    public static IValue GroupNormalization(IValue x, IValue scale, IValue bias,
        float? epsilon, long? numGroups, long? stashType = null)
        => NodeBuilder.BuildNodeSingleOut(GROUP_NORMALIZATION, [x, scale, bias], [
            (AttrEpsilon, epsilon),
            (AttrNumGroups, numGroups),
            (AttrStashType, stashType)]);

    public static IValue Gemm(IValue a, IValue b, IValue? c = null,
        float? alpha = null, float? beta = null, long? transA = null, long? transB = null)
        => NodeBuilder.BuildNodeSingleOut(GEMM, [a, b, c], [
            (AttrAlpha, alpha),
            (AttrBeta, beta),
            (AttrTransA, transA),
            (AttrTransB, transB)]);

    public static IValue InstanceNormalization(IValue x, IValue scale, IValue bias,
        float? epsilon = null)
        => NodeBuilder.BuildNodeSingleOut(INSTANCE_NORMALIZATION, [x, scale, bias], [
            (AttrEpsilon, epsilon)]);
}