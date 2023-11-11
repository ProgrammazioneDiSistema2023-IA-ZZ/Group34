use crate::OnnxError;
use crate::onnx::{TensorProto, NodeProto};
use crate::operations::exp::exp;
use crate::operations::reducesum::reducesum;
use crate::operations::utils::{
    convert_to_output_tensor, stack_along_batch_dimension, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;

/// `softmax` - ONNX Node Implementation for Softmax Normalization
///
/// The `softmax` function computes the normalized exponential values for the input tensor. It essentially
/// scales the input tensor's values so that they fall between 0 and 1, which makes them interpretable as
/// probabilities. Mathematically, it is represented as:
///
/// Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
///
/// # Arguments
///
/// * `input` - Reference to the input tensor.
/// * `node` - A reference to the ONNX NodeProto, housing attributes specific to the Softmax operation.
///
/// # Returns
///
/// * `Result<TensorProto, OnnxError>` - Outputs the tensor containing Softmax-normalized values. If there's an
///   error during the Softmax computation, an error (`OnnxError`) will be returned.
///
/// # Attributes
///
/// * `axis` - Specifies the dimension along which the Softmax computation should be performed. It defaults
///   to `-1`. A negative value denotes that the counting of dimensions should be from the back. Valid range
///   is given by `[-r, r-1]`, where `r` denotes the rank of the input tensor.
///
/// # Notes
///
/// The resultant tensor post-Softmax computation retains the original shape. The values within this tensor represent
/// the Softmax values of the corresponding input tensor elements.
pub fn softmax(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let exp_nd_array = tensor_proto_to_ndarray::<f32>(&exp(input, node)?).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    let reduce_nd_array = tensor_proto_to_ndarray::<f32>(&reducesum(&exp(input, node)?, node)?)
        .map_err(|_| {
            OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
        })?;

    let batch_size = exp_nd_array.shape()[0];
    let mut result_list = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let batch_exp = exp_nd_array.index_axis(Axis(0), b);

        let batch_sum = reduce_nd_array.index_axis(Axis(0), b);
        let batch_result = batch_exp.to_owned() / batch_sum.to_owned();
        result_list.push(batch_result);
    }

    let result = stack_along_batch_dimension(result_list)?;

    convert_to_output_tensor(node, result)
}