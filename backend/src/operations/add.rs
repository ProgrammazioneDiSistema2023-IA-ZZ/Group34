use crate::{operations::utils::{convert_to_output_tensor, tensor_proto_to_ndarray}, onnx::{TensorProto, NodeProto}, utils::OnnxError};
use ndarray::prelude::*;
#[allow(irrefutable_let_patterns)]
pub fn add(
    inputs: Vec<TensorProto>,
    initializers: Vec<TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Converti i tensori di input in ndarray
    let inputs_nd_array = inputs
        .iter()
        .map(|x| {
            tensor_proto_to_ndarray::<f32>(x).map_err(|_| {
                OnnxError::ConversionError("Failed to convert TensorProto to ndarray".to_string())
            })
        })
        .collect::<Result<Vec<ArrayD<f32>>, OnnxError>>()?;

    // Unisci i tensori di input e inizializzazione
    let mut merged_tensors = Vec::new();
    merged_tensors.extend(inputs_nd_array);

    if let param_tensors = initializers {
        let initializers_nd_array = param_tensors
            .iter()
            .map(|x| {
                tensor_proto_to_ndarray::<f32>(x).map_err(|_| {
                    OnnxError::ConversionError(
                        "Failed to convert TensorProto to ndarray".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<ArrayD<f32>>, OnnxError>>()?;

        merged_tensors.extend(initializers_nd_array);
    }

    // Verifica se ci sono tensori da aggiungere
    if merged_tensors.is_empty() {
        return Err(OnnxError::MissingInput(
            "Input mancanti per l'operazione di addizione".to_string(),
        ));
    }

    // Esegui l'operazione di addizione
    let result = merged_tensors
        .iter()
        .skip(1)
        .fold(merged_tensors[0].clone(), |acc, x| acc + x);

    // Converte il risultato in formato TensorProto
    convert_to_output_tensor(node, result)
}