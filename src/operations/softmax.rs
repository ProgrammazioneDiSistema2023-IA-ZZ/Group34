use crate::OnnxError;
use crate::onnx::{TensorProto, NodeProto};
use crate::operations::exp::exp;
use crate::operations::reducesum::reducesum;
use crate::operations::utils::{
    convert_to_output_tensor, stack_along_batch_dimension, tensor_proto_to_ndarray,
};
use ndarray::prelude::*;
pub fn softmax(input: &TensorProto, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    // Calcola l'esponenziale di ciascun elemento del tensore di input.
    let exp_nd_array = tensor_proto_to_ndarray::<f32>(&exp(input, node)?).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Calcola la somma degli elementi esponenziali.
    let reduce_nd_array = tensor_proto_to_ndarray::<f32>(&reducesum(&exp(input, node)?, node)?)
        .map_err(|_| {
            OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
        })?;

    // Ottieni la dimensione del batch.
    let batch_size = exp_nd_array.shape()[0];
    let mut result_list = Vec::with_capacity(batch_size);

    // Itera su ogni batch.
    for b in 0..batch_size {
        // Estrai il batch corrente.
        let batch_exp = exp_nd_array.index_axis(Axis(0), b);
        let batch_sum = reduce_nd_array.index_axis(Axis(0), b);

        // Calcola la softmax per il batch corrente.
        let batch_result = batch_exp.to_owned() / batch_sum.to_owned();
        result_list.push(batch_result);
    }

    // Combina i risultati lungo l'asse del batch.
    let result = stack_along_batch_dimension(result_list)?;

    // Converti il risultato in un TensorProto e restituisci.
    convert_to_output_tensor(node, result)
}
