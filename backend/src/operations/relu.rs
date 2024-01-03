use crate::{operations::utils::tensor_proto_to_ndarray, onnx::{TensorProto, NodeProto}, utils::OnnxError};
use ndarray::ArrayD;

use super::utils::convert_to_output_tensor;

pub fn relu(input: Vec<TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap();//c'è solo un input
    // Converti TensorProto in ndarray.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Applica la funzione ReLU a ciascun elemento del tensor.
    let relu_values: Vec<f32> = input_nd_array
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();

    // Creazione di un nuovo ArrayD dai valori ReLU calcolati.
    let result = ArrayD::from_shape_vec(input_nd_array.raw_dim(), relu_values)
        .map_err(|_| OnnxError::ShapeMismatch("Failed to reshape!".into()))?;

    // Converti il risultato finale in TensorProto e restituisci.
    convert_to_output_tensor(node, result)
}
