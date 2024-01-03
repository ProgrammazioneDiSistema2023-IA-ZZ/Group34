use crate::{operations::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, tensor_proto_to_ndarray,}, onnx::{NodeProto, TensorProto}, utils::OnnxError};
use ndarray::*;
pub fn concat(inputs: &Vec<TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    // Estrai gli attributi del nodo.
    let attributes = extract_attributes(&node.attribute)?;
    let axis = get_int_attribute(&attributes, "axis", None)? as usize;

    // Converti i TensorProto in ndarrays.
    let inputs_nd_array: Vec<_> = inputs
        .iter()
        .map(|tp| tensor_proto_to_ndarray::<f32>(tp).unwrap())
        .collect();

    // Concatena i tensori lungo l'asse specificato.
    let result = concat_tensors(inputs_nd_array, axis)?;

    // Converte il risultato in formato TensorProto.
    convert_to_output_tensor(node, result)
}

// Concatena i tensori lungo l'asse specificato.
// Questa funzione accetta un vettore di ndarrays e un indice di asse lungo il quale eseguire la concatenazione.

fn concat_tensors<T, D>(
    tensors: Vec<ArrayBase<OwnedRepr<T>, D>>,
    axis: usize,
) -> Result<ArrayBase<OwnedRepr<T>, D>, OnnxError>
where
    T: Clone,
    D: Dimension + RemoveAxis,
{
    // Verifica se i tensori hanno lo stesso numero di dimensioni.
    let first_dim = tensors[0].ndim();

    if tensors.iter().any(|tensor| tensor.ndim() != first_dim) {
        return Err(OnnxError::ShapeError(
            "All tensors must have the same number of dimensions.".to_string(),
        ));
    }

    // Verifica se l'asse specificato è valido.
    if axis >= first_dim {
        return Err(OnnxError::ShapeError(
            "Specified axis is out of bounds for the given tensors.".to_string(),
        ));
    }

    // Ottieni le viste degli ndarrays.
    let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();

    // Esegui la concatenazione lungo l'asse specificato.
    let concatenated_output = concatenate(Axis(axis), &views).map_err(|_| {
        OnnxError::ShapeError("Failed to concatenate tensors along specified axis.".to_string())
    })?;

    Ok(concatenated_output)
}