use crate::{
    onnx::{NodeProto, TensorProto},
    operations::gemm::gemm, utils::OnnxError,
};

pub fn matmul(
    inputs: Vec<TensorProto>,
    initializers: Vec<TensorProto>,
    node: &NodeProto,
    is_par_enabled: bool,
) -> Result<TensorProto, OnnxError> {
    gemm(inputs, initializers, node, is_par_enabled)
}
