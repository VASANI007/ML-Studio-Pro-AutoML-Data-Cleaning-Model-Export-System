def export_onnx(model, X_sample):
    try:
        if X_sample is None:
            return None

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import numpy as np

        X_sample = X_sample.astype(np.float32)

        initial_type = [('input', FloatTensorType([None, X_sample.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        return onnx_model.SerializeToString()

    except Exception as e:
        return None