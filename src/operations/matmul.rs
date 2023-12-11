use crate::{operations::gemm::gemm, onnx::{TensorProto, NodeProto}, OnnxError};

pub fn matmul(
    inputs: Vec<TensorProto>,
    initializers:Vec<TensorProto>,
    node: &NodeProto,
    flag:  bool,
) -> Result<TensorProto, OnnxError> {
    gemm(inputs, initializers, node,flag)
}