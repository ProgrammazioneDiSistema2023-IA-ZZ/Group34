use crate::{operations::gemm::gemm, onnx::{TensorProto, NodeProto}, OnnxError};

pub fn matmul(
    inputs: Vec<TensorProto>,
    initializers:Vec<TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    gemm(inputs, initializers, node)
}