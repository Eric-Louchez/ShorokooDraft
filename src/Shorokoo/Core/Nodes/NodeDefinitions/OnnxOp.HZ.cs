using Shorokoo;
using Shorokoo.Core;
using Shorokoo.Core.Nodes;
using Shorokoo.Core.Nodes.AutoDiff;
using Shorokoo.Core.Training;
using Shorokoo.Modules;
using Shorokoo.Graph;
using Shorokoo.Core.Nodes.OnnxNodes;
using System.Collections.Immutable;
using static Shorokoo.Core.Nodes.NodeDefinitions.OnnxOpAttributeNames;
using static Shorokoo.Core.Nodes.NodeDefinitions.OpCodes;

namespace Shorokoo.Core.Nodes.NodeDefinitions;

public static partial class OnnxOp
{
    public static IValue Identity(IValue x, int? rank)
        => NodeBuilder.BuildNodeSingleOut(IDENTITY, [x], [(InternalAttrRank, (long?)rank)]);

    public static Node LoopOpen(IValue? maxNumIterations, IValue? condition, IValue?[] loopVariableInitializers)
        => NodeBuilder.BuildNode(LOOP_OPEN, [maxNumIterations, condition, .. loopVariableInitializers], []);

    public static Node LoopClose(IValue? continueWhile, IValue[] loopVariableUpdaters, IValue[] scanVariableInputs, Node openNode)
        => NodeBuilder.BuildNode(LOOP_CLOSE, [], [(AttrBody, (IValue?[])[continueWhile, .. loopVariableUpdaters, .. scanVariableInputs])], openNode: openNode);

    public static Node IfOpen(IValue condition)
        => NodeBuilder.BuildNode(IF_OPEN, [condition], []);

    public static IValue[] IfClose(IValue[] thenBranch, IValue[] elseBranch, Node openNode)
     => NodeBuilder.BuildNodeMultiOut(IF_CLOSE, [], [(AttrThenBranch, thenBranch), (AttrElseBranch, elseBranch)], openNode: openNode);

    public static IValue LeakyRelu(IValue x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(LEAKY_RELU, [x], [(AttrAlpha, alpha)]);

    public static IValue Less(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(LESS, [left, right], []);

    public static IValue LessOrEqual(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(LESS_OR_EQUAL, [left, right], []);

    public static IValue LoopIndexVariable()
        => NodeBuilder.BuildNodeSingleOut(LOOP_INDEX_VARIABLE, [], []);

    public static IValue LoopFakeInput(DType type, int? rank, DataStructure structure)
        => NodeBuilder.BuildNodeSingleOut(LOOP_FAKE_INPUT, [], [(AttrDtype, type), (InternalAttrRank, (long?)rank), (InternalAttrStructure, structure)]);

    public static IValue LoopScanZombie(IValue scannee)
        => NodeBuilder.BuildNodeSingleOut(LOOP_SCAN_VARIABLE, [scannee], []);
    
    public static IValue Log(IValue x)
        => NodeBuilder.BuildNodeSingleOut(LOG, [x], []);

    public static IValue LpPool(IValue x, AutoPad? autoPad, bool? ceilMode,
        long[]? dilations, long[] kernelShape, long? p, long[]? pads, long[]? strides)
        => NodeBuilder.BuildNodeSingleOut(LP_POOL, [x], [
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrP, p),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IValue MatMul(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(MATMUL, [left, right], []);

    public static IValue MatMulInteger(IValue a, IValue b, IValue? aZeroPoint = null, IValue? bZeroPoint = null)
        => NodeBuilder.BuildNodeSingleOut(MATMUL_INTEGER, [a, b, aZeroPoint, bZeroPoint], []);

    public static IValue Max(params IValue[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MAX, inputs, []);

    public static IValue Mean(params IValue[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MEAN, inputs, []);

    public static IValue MaxPool(IValue x, AutoPad? autoPad = null, bool? ceilMode = null,
        long[]? dilations = null, long[]? kernelShape = null, long[]? pads = null,
        long? storageOrder = null, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(MAX_POOL, [x], [
            (InternalAttrHasOptionalOutputs, false),
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStorageOrder, storageOrder),
            (AttrStrides, strides)]);

    public static (IValue y, IValue indices) MaxPoolWithIndices(IValue x, AutoPad? autoPad = null, bool? ceilMode = null,
        long[]? dilations = null, long[]? kernelShape = null, long[]? pads = null,
        long? storageOrder = null, long[]? strides = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(MAX_POOL, [x], [
            (InternalAttrHasOptionalOutputs, true),
            (AttrAutoPad, autoPad),
            (AttrCeilMode, ceilMode),
            (AttrDilations, dilations),
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStorageOrder, storageOrder),
            (AttrStrides, strides)]);
        return (retval[0], retval[1]);
    }

    public static IValue MaxUnpool(IValue x, IValue indices,
        long[] kernelShape, long[]? pads = null, long[]? strides = null, IValue? outputShape = null)
        // The optional output_shape input is only emitted when present — ORT's MaxUnpool
        // kernel rejects a trailing empty-name optional input ("input count mismatch").
        => NodeBuilder.BuildNodeSingleOut(MAX_UNPOOL,
            outputShape is null ? [x, indices] : [x, indices, outputShape], [
            (AttrKernelShape, kernelShape),
            (AttrPads, pads),
            (AttrStrides, strides)]);

    public static IValue MaxRoiPool(IValue x, IValue rois,
        long[] pooledShape, float? spatialScale = null)
        => NodeBuilder.BuildNodeSingleOut(MAX_ROI_POOL, [x, rois], [
            (AttrPooledShape, pooledShape),
            (AttrSpatialScale, spatialScale)]);

    public static IValue Min(params IValue[] inputs)
        => NodeBuilder.BuildNodeSingleOut(MIN, inputs, []);

    public static IValue Mod(IValue a, IValue b, bool? fmod = null)
        => NodeBuilder.BuildNodeSingleOut(MOD, [a, b], [(AttrFmod, fmod)]);

    public static IValue Mul(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(MUL, [left, right], []);

    public static IValue Neg(IValue x)
        => NodeBuilder.BuildNodeSingleOut(NEG, [x], []);

    public static IValue NonMaxSuppression(IValue boxes, IValue scores,
        IValue? maxOutputBoxesPerClass = null, IValue? iouThreshold = null,
        IValue? scoreThreshold = null, bool? centerPointBox = null)
        => NodeBuilder.BuildNodeSingleOut(NON_MAX_SUPPRESSION, 
            [boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold],
            [(AttrCenterPointBox, centerPointBox)]);

    public static IValue NonZero(IValue x)
        => NodeBuilder.BuildNodeSingleOut(NON_ZERO, [x], []);

    public static IValue Not(IValue x)
        => NodeBuilder.BuildNodeSingleOut(NOT, [x], []);

    public static IValue Optional(IValue? x, DataStructure structure, DType type)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL, x is null ? [] : [x], [(AttrType, (structure, type))]);

    public static IValue OptionalGetElement(IValue x)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL_GET_ELEMENT, [x], []);

    public static IValue OptionalHasElement(IValue x)
        => NodeBuilder.BuildNodeSingleOut(OPTIONAL_HAS_ELEMENT, [x], []);

    public static IValue Or(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(OR, [left, right], []);

    public static IValue Pad(IValue data, IValue pads, IValue? constantValue = null,
        IValue? axes = null, PadMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(PAD, [data, pads, constantValue, axes],
            [(AttrMode, mode)]);

    public static IValue Pow(IValue x, IValue y)
        => NodeBuilder.BuildNodeSingleOut(POW, [x, y], []);

    public static IValue Range(IValue start, IValue limit, IValue delta)
        => NodeBuilder.BuildNodeSingleOut(RANGE, [start, limit, delta], []);

    public static IValue RandomNormal(long[] shape, float? mean = null, float? scale = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_NORMAL, [], [
            (AttrShape, shape), (AttrDtype, dtype), (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    public static IValue RandomNormalLike(IValue input, float? mean = null, float? scale = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_NORMAL_LIKE, [input], [
            (AttrDtype, dtype), (AttrMean, mean), (AttrScale, scale), (AttrSeed, seed)]);

    public static IValue RandomUniform(long[] shape, float? high = null, float? low = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_UNIFORM, [], [
            (AttrShape, shape), (AttrDtype, dtype), (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    public static IValue RandomUniformLike(IValue input, float? high = null, float? low = null, DType? dtype = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(RANDOM_UNIFORM_LIKE, [input], [
            (AttrDtype, dtype), (AttrHigh, high), (AttrLow, low), (AttrSeed, seed)]);

    public static IValue Reciprocal(IValue x)
        => NodeBuilder.BuildNodeSingleOut(RECIPROCAL, [x], []);

    public static IValue ReduceL1(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_L1, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceL2(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_L2, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceLogSum(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_LOG_SUM, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceLogSumExp(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_LOG_SUM_EXP, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceMax(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MAX, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceMean(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MEAN, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceMin(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_MIN, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceProd(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_PROD, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceSum(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_SUM, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static IValue ReduceSumSquare(IValue data, IValue? axes = null,
        bool? keepdims = null, bool? noopWithEmptyAxes = null)
        => NodeBuilder.BuildNodeSingleOut(REDUCE_SUM_SQUARE, [data, axes],
            [(AttrKeepdims, keepdims), (AttrNoopWithEmptyAxes, noopWithEmptyAxes)]);

    public static (IValue y, IValue yH) Rnn(IValue x, IValue w, IValue r,
        IValue? b, IValue? sequenceLens, IValue? initialH,
        float[]? activationAlpha, float[]? activationBeta, string[]? activations,
        float? clip, RNNDirection? direction, long? hiddenSize,
        bool? layout)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(RNN, [x, w, r, b, sequenceLens, initialH], [
            (AttrActivationAlpha, activationAlpha),
            (AttrActivationBeta, activationBeta),
            (AttrActivations, activations),
            (AttrClip, clip),
            (AttrDirection, direction),
            (AttrHiddenSize, hiddenSize),
            (AttrLayout, layout)]);
        return (retval[0], retval[1]);
    }

    public static IValue Relu(IValue x)
        => NodeBuilder.BuildNodeSingleOut(RELU, [x], []);

    public static IValue Reshape(IValue data, IValue shape, bool allowZero)
        => NodeBuilder.BuildNodeSingleOut(RESHAPE, [data, shape], [(AttrAllowzero, allowZero)]);

    public static IValue ReverseSequence(IValue input, IValue sequenceLens, long? batchAxis = null, long? timeAxis = null)
        => NodeBuilder.BuildNodeSingleOut(REVERSE_SEQUENCE, [input, sequenceLens], [(AttrBatchAxis, batchAxis), (AttrTimeAxis, timeAxis)]);

    public static IValue Resize(IValue x, IValue? roi, IValue? scales,
        IValue? sizes, bool? antialias, long[]? axes,
        CoordinateTransformationMode? coordinateTransformationMode,
        float? cubicCoeffA, bool? excludeOutside,
        float? extrapolationValue, KeepAspectRatioPolicy? keepAspectRatioPolicy,
        ResizeMode? mode, NearestMode? nearestMode)
        => NodeBuilder.BuildNodeSingleOut(RESIZE, [x, roi, scales, sizes], [
            (AttrAntialias, antialias),
            (AttrAxes, axes),
            (AttrCoordinateTransformationMode, coordinateTransformationMode),
            (AttrCubicCoeffA, cubicCoeffA),
            (AttrExcludeOutside, excludeOutside),
            (AttrExtrapolationValue, extrapolationValue),
            (AttrKeepAspectRatioPolicy, keepAspectRatioPolicy),
            (AttrMode, mode),
            (AttrNearestMode, nearestMode)]);

    public static IValue RoiAlign(IValue x, IValue rois, IValue batchIndices,
        RoiAlignTransformationMode? coordinateTransformationMode = null,
        RoiAlignMode? mode = null, long? outputHeight = null, long? outputWidth = null,
        long? samplingRatio = null, float? spatialScale = null)
        => NodeBuilder.BuildNodeSingleOut(ROI_ALIGN, [x, rois, batchIndices], [
            (AttrCoordinateTransformationMode, coordinateTransformationMode),
            (AttrMode, mode),
            (AttrOutputHeight, outputHeight),
            (AttrOutputWidth, outputWidth),
            (AttrSamplingRatio, samplingRatio),
            (AttrSpatialScale, spatialScale)]);

    public static IValue ScatterElements(IValue data, IValue indices, IValue updates,
        long? axis = null, ScatterNDReduction? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(SCATTER_ELEMENTS, [data, indices, updates],
            [(AttrAxis, axis), (AttrReduction, reduction)]);

    public static IValue ScatterND(IValue data, IValue indices, IValue updates,
        ScatterNDReduction? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(SCATTER_ND, [data, indices, updates],
            [(AttrReduction, reduction)]);

    public static IValue Selu(IValue x, float? alpha = null, float? gamma = null)
        => NodeBuilder.BuildNodeSingleOut(SELU, [x], [(AttrAlpha, alpha), (AttrGamma, gamma)]);

    public static IValue SequenceAt(IValue input, IValue position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_AT, [input, position], []);

    public static IValue SequenceConstruct(params IValue[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONSTRUCT, inputs, []);

    internal static IValue SequenceConstruct(Function targetFunction, params IValue[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_CONSTRUCT, inputs, [], targetFunction: targetFunction);

    public static IValue SequenceEmpty(DType type, Function? targetFunction = null)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_EMPTY, [], [(AttrDtype, type)], targetFunction: targetFunction);

    public static IValue SequenceErase(IValue input, IValue? position = null)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_ERASE, [input, position], []);

    public static IValue SequenceErase(Function targetFunction, IValue input, IValue? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_ERASE, [input, position], [], targetFunction: targetFunction);

    public static IValue SequenceInsert(IValue input, IValue tensor, IValue? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_INSERT, [input, tensor, position], []);

    public static IValue SequenceInsert(Function targetFunction, IValue input, IValue tensor, IValue? position)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_INSERT, [input, tensor, position], [], targetFunction: targetFunction);

    public static IValue SequenceLength(IValue input)
        => NodeBuilder.BuildNodeSingleOut(SEQUENCE_LENGTH, [input], []);

    public static IValue Shape(IValue data, long? end = null, long? start = null)
        => NodeBuilder.BuildNodeSingleOut(SHAPE, [data], [(AttrEnd, end), (AttrStart, start)]);

    public static IValue Sigmoid(IValue x)
        => NodeBuilder.BuildNodeSingleOut(SIGMOID, [x], []);

    public static IValue Sign(IValue x)
        => NodeBuilder.BuildNodeSingleOut(SIGN, [x], []);

    public static IValue Sin(IValue num)
        => NodeBuilder.BuildNodeSingleOut(SIN, [num], []);

    public static IValue Sinh(IValue num)
        => NodeBuilder.BuildNodeSingleOut(SINH, [num], []);

    public static IValue Slice(IValue data, IValue starts, IValue ends,
        IValue? axes = null, IValue? steps = null)
        => NodeBuilder.BuildNodeSingleOut(SLICE, [data, starts, ends, axes, steps], []);

    public static IValue Softmax(IValue input, long? axis)
        => NodeBuilder.BuildNodeSingleOut(SOFTMAX, [input], [(AttrAxis, axis)]);

    public static IValue[] Split(IValue data, IValue? splits, long? axis, long? numOutputs, long variadicOutputCount)
        => NodeBuilder.BuildNodeMultiOut(SPLIT, [data, splits], [(AttrAxis, axis), (AttrNumOutputs, numOutputs)],
            outputNames: Enumerable.Repeat((string?)null, (int)variadicOutputCount).ToArray());

    public static IValue Sqrt(IValue num)
        => NodeBuilder.BuildNodeSingleOut(SQRT, [num], []);

    public static IValue Squeeze(IValue data, IValue? axes)
        => NodeBuilder.BuildNodeSingleOut(SQUEEZE, [data, axes], []);

    public static IValue Sub(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(SUB, [left, right], []);

    public static IValue Sum(params IValue[] inputs)
        => NodeBuilder.BuildNodeSingleOut(SUM, inputs, []);

    public static IValue Tan(IValue x)
        => NodeBuilder.BuildNodeSingleOut(TAN, [x], []);

    public static IValue Tanh(IValue x)
        => NodeBuilder.BuildNodeSingleOut(TANH, [x], []);

    public static IValue Tile(IValue input, IValue repeats)
        => NodeBuilder.BuildNodeSingleOut(TILE, [input, repeats], []);

    public static (IValue values, IValue indices) TopK(IValue x, IValue k,
        long? axis = null, bool? largest = null, bool? sorted = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(TOPK, [x, k], [
            (AttrAxis, axis),
            (AttrLargest, largest),
            (AttrSorted, sorted)]);
        return (retval[0], retval[1]);
    }

    public static IValue Transpose(IValue data, long[]? perm = null)
        => NodeBuilder.BuildNodeSingleOut(TRANSPOSE, [data], [(AttrPerm, perm)]);

    public static (IValue y, IValue indices, IValue inverseIndices, IValue counts) Unique(
        IValue x, long? axis = null, bool? sorted = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(UNIQUE, [x], [
            (AttrAxis, axis),
            (AttrSorted, sorted)]);
        return (retval[0], retval[1], retval[2], retval[3]);
    }

    public static IValue Unsqueeze(IValue data, IValue axes)
        => NodeBuilder.BuildNodeSingleOut(UNSQUEEZE, [data, axes], []);

    public static IValue Where(IValue condition, IValue x, IValue y)
        => NodeBuilder.BuildNodeSingleOut(WHERE, [condition, x, y], []);

    public static IValue Xor(IValue left, IValue right)
        => NodeBuilder.BuildNodeSingleOut(XOR, [left, right], []);

    public static IValue SpaceToDepth(IValue input, long? blockSize)
        => NodeBuilder.BuildNodeSingleOut(SPACE_TO_DEPTH, [input], [(AttrBlocksize, blockSize)]);

    public static IValue Trilu(IValue input, IValue? k = null, long? upper = null)
        => NodeBuilder.BuildNodeSingleOut(TRILU, [input, k], [(AttrUpper, upper)]);

    public static IValue LpNormalization(IValue input, long? axis = null, long? p = null)
        => NodeBuilder.BuildNodeSingleOut(LP_NORMALIZATION, [input], [(AttrAxis, axis), (AttrP, p)]);

    public static (IValue y, IValue yH, IValue yC) Lstm(IValue x, IValue w, IValue r,
        IValue? b, IValue? sequenceLens, IValue? initialH, IValue? initialC, IValue? p,
        float[]? activationAlpha, float[]? activationBeta, string[]? activations,
        float? clip, LSTMDirection? direction, long? hiddenSize,
        bool? inputForget, bool? layout)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(LSTM, [x, w, r, b, sequenceLens, initialH, initialC, p], [
            (AttrActivationAlpha, activationAlpha),
            (AttrActivationBeta, activationBeta),
            (AttrActivations, activations),
            (AttrClip, clip),
            (AttrDirection, direction),
            (AttrHiddenSize, hiddenSize),
            (AttrInputForget, inputForget),
            (AttrLayout, layout)]);
        return (retval[0], retval[1], retval[2]);
    }

    public static IValue Lrn(IValue x, float? alpha = null, float? beta = null,
        float? bias = null, long? size = null)
        => NodeBuilder.BuildNodeSingleOut(LRN, [x], [
            (AttrAlpha, alpha),
            (AttrBeta, beta),
            (AttrBias, bias),
            (AttrSize, size)]);

    public static IValue Upsample(IValue x, IValue scales, ResizeMode? mode = null)
        => NodeBuilder.BuildNodeSingleOut(UPSAMPLE, [x, scales], [(AttrMode, mode)]);

    // -- New opset-21 operators ---------------------------------------------

    public static IValue Hardmax(IValue input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(HARDMAX, [input], [(AttrAxis, axis)]);

    public static IValue HardSigmoid(IValue x, float? alpha = null, float? beta = null)
        => NodeBuilder.BuildNodeSingleOut(HARD_SIGMOID, [x], [(AttrAlpha, alpha), (AttrBeta, beta)]);

    public static IValue HardSwish(IValue x)
        => NodeBuilder.BuildNodeSingleOut(HARD_SWISH, [x], []);

    public static IValue HammingWindow(IValue size, DType? outputDatatype = null, bool? periodic = null)
        => NodeBuilder.BuildNodeSingleOut(HAMMING_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static IValue HannWindow(IValue size, DType? outputDatatype = null, bool? periodic = null)
        => NodeBuilder.BuildNodeSingleOut(HANN_WINDOW, [size], [(AttrOutputDatatype, outputDatatype), (AttrPeriodic, periodic)]);

    public static IValue ImageDecoder(IValue encodedStream, string? pixelFormat = null)
        => NodeBuilder.BuildNodeSingleOut(IMAGE_DECODER, [encodedStream], [(AttrPixelFormat, pixelFormat)]);

    public static IValue IsInf(IValue x, bool? detectNegative = null, bool? detectPositive = null)
        => NodeBuilder.BuildNodeSingleOut(IS_INF, [x], [(AttrDetectNegative, detectNegative), (AttrDetectPositive, detectPositive)]);

    public static IValue IsNaN(IValue x)
        => NodeBuilder.BuildNodeSingleOut(IS_NAN, [x], []);

    public static (IValue y, IValue? mean, IValue? invStdDev) LayerNormalization(
        IValue x, IValue scale, IValue? b = null,
        long? axis = null, float? epsilon = null, long? stashType = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(LAYER_NORMALIZATION, [x, scale, b],
            [(AttrAxis, axis), (AttrEpsilon, epsilon), (AttrStashType, stashType)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null, retval.Length > 2 ? retval[2] : null);
    }

    public static IValue LogSoftmax(IValue input, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(LOG_SOFTMAX, [input], [(AttrAxis, axis)]);

    public static IValue MeanVarianceNormalization(IValue x, long[]? axes = null)
        => NodeBuilder.BuildNodeSingleOut(MEAN_VARIANCE_NORMALIZATION, [x], [(AttrAxes, axes)]);

    public static IValue MelWeightMatrix(
        IValue numMelBins, IValue dftLength, IValue sampleRate,
        IValue lowerEdgeHertz, IValue upperEdgeHertz, DType? outputDatatype = null)
        => NodeBuilder.BuildNodeSingleOut(MEL_WEIGHT_MATRIX,
            [numMelBins, dftLength, sampleRate, lowerEdgeHertz, upperEdgeHertz],
            [(AttrOutputDatatype, outputDatatype)]);

    public static IValue Mish(IValue x)
        => NodeBuilder.BuildNodeSingleOut(MISH, [x], []);

    public static IValue Multinomial(IValue input, DType? dtype = null, long? sampleSize = null, float? seed = null)
        => NodeBuilder.BuildNodeSingleOut(MULTINOMIAL, [input],
            [(AttrDtype, dtype), (AttrSampleSize, sampleSize), (AttrSeed, seed)]);

    public static IValue NegativeLogLikelihoodLoss(
        IValue input, IValue target, IValue? weight = null,
        long? ignoreIndex = null, string? reduction = null)
        => NodeBuilder.BuildNodeSingleOut(NEGATIVE_LOG_LIKELIHOOD_LOSS, [input, target, weight],
            [(AttrIgnoreIndex, ignoreIndex), (AttrReduction, reduction)]);

    public static IValue OneHot(IValue indices, IValue depth, IValue values, long? axis = null)
        => NodeBuilder.BuildNodeSingleOut(ONE_HOT, [indices, depth, values], [(AttrAxis, axis)]);

    public static IValue PRelu(IValue x, IValue slope)
        => NodeBuilder.BuildNodeSingleOut(P_RELU, [x, slope], []);

    public static IValue QuantizeLinear(
        IValue x, IValue yScale, IValue? yZeroPoint = null,
        long? axis = null, long? blockSize = null, DType? outputDatatype = null,
        bool? saturate = null, long? precision = null)
        => NodeBuilder.BuildNodeSingleOut(QUANTIZE_LINEAR, [x, yScale, yZeroPoint],
            [(AttrAxis, axis), (AttrBlockSize, blockSize), (AttrOutputDtype, outputDatatype),
             (AttrSaturate, saturate), (AttrPrecision, precision)]);

    public static IValue QLinearMatMul(
        IValue a, IValue aScale, IValue aZeroPoint,
        IValue b, IValue bScale, IValue bZeroPoint,
        IValue yScale, IValue yZeroPoint)
        => NodeBuilder.BuildNodeSingleOut(QLINEAR_MATMUL,
            [a, aScale, aZeroPoint, b, bScale, bZeroPoint, yScale, yZeroPoint], []);

    public static IValue QLinearConv(
        IValue x, IValue xScale, IValue xZeroPoint,
        IValue w, IValue wScale, IValue wZeroPoint,
        IValue yScale, IValue yZeroPoint, IValue? b = null,
        AutoPad? autoPad = null, long[]? dilations = null, long? group = null,
        long[]? kernelShape = null, long[]? pads = null, long[]? strides = null)
        => NodeBuilder.BuildNodeSingleOut(QLINEAR_CONV,
            [x, xScale, xZeroPoint, w, wScale, wZeroPoint, yScale, yZeroPoint, b],
            [(AttrAutoPad, autoPad), (AttrDilations, dilations), (AttrGroup, group),
             (AttrKernelShape, kernelShape), (AttrPads, pads), (AttrStrides, strides)]);

    public static IValue RegexFullMatch(IValue x, string? pattern = null)
        => NodeBuilder.BuildNodeSingleOut(REGEX_FULL_MATCH, [x], [(AttrPattern, pattern)]);

    public static IValue Round(IValue x)
        => NodeBuilder.BuildNodeSingleOut(ROUND, [x], []);

    public static IValue Shrink(IValue input, float? bias = null, float? lambd = null)
        => NodeBuilder.BuildNodeSingleOut(SHRINK, [input], [(AttrBias, bias), (AttrLambd, lambd)]);

    public static IValue Size(IValue data)
        => NodeBuilder.BuildNodeSingleOut(SIZE, [data], []);

    public static IValue Softplus(IValue x)
        => NodeBuilder.BuildNodeSingleOut(SOFTPLUS, [x], []);

    public static IValue Softsign(IValue input)
        => NodeBuilder.BuildNodeSingleOut(SOFTSIGN, [input], []);

    public static (IValue output, IValue? logProb) SoftmaxCrossEntropyLoss(
        IValue scores, IValue labels, IValue? weights = null,
        long? ignoreIndex = null, string? reduction = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(SOFTMAX_CROSS_ENTROPY_LOSS, [scores, labels, weights],
            [(AttrIgnoreIndex, ignoreIndex), (AttrReduction, reduction)]);
        return (retval[0], retval.Length > 1 ? retval[1] : null);
    }

    public static IValue SplitToSequence(IValue input, IValue? split = null, long? axis = null, long? keepdims = null)
        => NodeBuilder.BuildNodeSingleOut(SPLIT_TO_SEQUENCE, [input, split],
            [(AttrAxis, axis), (AttrKeepdims, keepdims)]);

    public static IValue STFT(
        IValue signal, IValue frameStep, IValue? window = null, IValue? frameLength = null,
        bool? onesided = null)
        => NodeBuilder.BuildNodeSingleOut(OpCodes.STFT, [signal, frameStep, window, frameLength],
            [(AttrOnesided, onesided)]);

    public static IValue StringConcat(IValue x, IValue y)
        => NodeBuilder.BuildNodeSingleOut(STRING_CONCAT, [x, y], []);

    public static IValue StringNormalizer(IValue x,
        string? caseChangeAction = null, long? isCaseSensitive = null,
        string? locale = null, string[]? stopwords = null)
        => NodeBuilder.BuildNodeSingleOut(STRING_NORMALIZER, [x],
            [(AttrCaseChangeAction, caseChangeAction), (AttrIsCaseSensitive, isCaseSensitive),
             (AttrLocale, locale), (AttrStopwords, stopwords)]);

    public static (IValue y, IValue numSplits) StringSplit(IValue x, string? delimiter = null, long? maxsplit = null)
    {
        var retval = NodeBuilder.BuildNodeMultiOut(STRING_SPLIT, [x],
            [(AttrDelimiter, delimiter), (AttrMaxsplit, maxsplit)]);
        return (retval[0], retval[1]);
    }

    public static IValue TfIdfVectorizer(IValue x,
        long? maxGramLength = null, long? maxSkipCount = null, long? minGramLength = null,
        string? mode = null, long[]? ngramCounts = null, long[]? ngramIndexes = null,
        long[]? poolInt64s = null, string[]? poolStrings = null, float[]? weights = null)
        => NodeBuilder.BuildNodeSingleOut(TFIDF_VECTORIZER, [x],
            [(AttrMaxGramLength, maxGramLength), (AttrMaxSkipCount, maxSkipCount),
             (AttrMinGramLength, minGramLength), (AttrMode, mode),
             (AttrNgramCounts, ngramCounts), (AttrNgramIndexes, ngramIndexes),
             (AttrPoolInt64s, poolInt64s), (AttrPoolStrings, poolStrings),
             (AttrWeights, weights)]);

    public static IValue ThresholdedRelu(IValue x, float? alpha = null)
        => NodeBuilder.BuildNodeSingleOut(THRESHOLDED_RELU, [x], [(AttrAlpha, alpha)]);

    /// <summary>Root-mean-square layer normalization over the suffix axes from <paramref name="axis"/> (ONNX RMSNormalization, opset 23+).
    /// Lowered inline to opset-21 primitives — <c>y = x / sqrt(mean(x², suffix axes) + epsilon) * scale</c>
    /// (ReduceMean/Sqrt/Div/Mul) — so the emitted ONNX stays at opset 21. The fused RMS_NORMALIZATION
    /// op definition and QEE kernel are retained; restore the fused emission here once a runtime
    /// registers it at a usable opset. (x and scale are assumed to share a dtype, the spec's common
    /// case.)</summary>
    public static IValue RMSNormalization(IValue x, IValue scale,
        long? axis = null, float? epsilon = null, long? stashType = null)
    {
        var a = axis ?? -1;
        IValue axesVar;
        if (a < 0)
        {
            // Negative axis: the suffix [axis, axis+1, ..., -1] is rank-independent.
            var count = (int)(-a);
            var axesArr = new long[count];
            for (var i = 0; i < count; i++) axesArr[i] = a + i;
            axesVar = Constant(axesArr);
        }
        else
        {
            // Non-negative axis: the suffix is [axis, ..., rank-1]. Compute it from the runtime
            // rank (Size of the shape vector) so it works on dynamic-rank inputs too.
            var rank = Size(Shape(x));
            axesVar = Range(Constant(a), rank, Constant(1L));
        }

        var meanSq = ReduceMean(Mul(x, x), axesVar, keepdims: true);
        var rms = Sqrt(Add(meanSq, CastLike(Constant(epsilon ?? 1e-5f), x, null)));
        return Mul(Div(x, rms), scale);
    }

    /// <summary>Rotary positional embedding (ONNX RotaryEmbedding, opset 23+); Y has X's shape.
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a faithful lowering of
    /// the fused op (position-id gather, interleaved vs half-split layouts, partial rotary dim) is
    /// intricate enough to belong in core (deferred core work) — so this
    /// throws rather than force a higher model opset. The ROTARY_EMBEDDING op definition and QEE kernel
    /// are retained; restore the fused emission here once a runtime supports it.</summary>
    public static IValue RotaryEmbedding(IValue x, IValue cosCache, IValue sinCache,
        IValue? positionIds = null, bool? interleaved = null, long? numHeads = null,
        long? rotaryEmbeddingDim = null)
        => throw new System.NotImplementedException(
            "RotaryEmbedding (ONNX opset 23) has no opset-21 equivalent, and Shorokoo emits a single " +
            "opset-21 model. A faithful lowering (position-id gather, interleaved/half-split layouts, " +
            "partial rotary dim) is deferred to the core project. The op " +
            "definition is retained; re-enable the fused emission here when a runtime supports it.");

    /// <summary>Swish activation y = x * sigmoid(alpha * x) (ONNX Swish, opset 24+).
    /// Lowered inline to opset-21 primitives (Mul/Sigmoid) so the emitted ONNX stays at opset 21 —
    /// ONNX Runtime 1.26 registers no Swish kernel on any provider. The fused SWISH op definition
    /// is retained; restore the fused emission here once a runtime supports it.</summary>
    public static IValue Swish(IValue x, float? alpha = null)
    {
        var a = alpha ?? 1.0f;
        var scaled = a == 1.0f ? x : Mul(x, CastLike(Constant(a), x, null));
        return Mul(x, Sigmoid(scaled));
    }

    /// <summary>Writes <paramref name="update"/> into <paramref name="pastCache"/> along the sequence axis at the per-batch write indices (ONNX TensorScatter, opset 24+).
    /// Not emittable today: Shorokoo exports a single opset-21 ONNX model, and a faithful lowering of
    /// the fused op (per-batch write indices, windowed/circular modes) is intricate enough to belong in
    /// core (deferred core work) — so this throws rather than force a
    /// higher model opset. The TENSOR_SCATTER op definition and (shape-only) QEE kernel are retained;
    /// restore the fused emission here once a runtime supports it.</summary>
    public static IValue TensorScatter(IValue pastCache, IValue update,
        IValue? writeIndices = null, long? axis = null, TensorScatterMode? mode = null)
        => throw new System.NotImplementedException(
            "TensorScatter (ONNX opset 24) has no opset-21 equivalent, and Shorokoo emits a single " +
            "opset-21 model. A faithful lowering (per-batch write indices, windowed/circular modes) is " +
            "deferred to the core project. The op definition is retained; " +
            "re-enable the fused emission here when a runtime supports it.");
}