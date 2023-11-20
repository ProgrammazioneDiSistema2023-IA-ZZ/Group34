use crate::{operations::utils::{
    convert_to_output_tensor, stack_along_batch_dimension, tensor_proto_to_ndarray,
}, OnnxError, onnx::{NodeProto, TensorProto}};
use ndarray::prelude::*;
// Funzione pubblica per implementare l'operazione di esponenziale in un grafo ONNX.
// L'operazione di esponenziale calcola l'esponenziale di ciascun elemento del tensore di input.
pub fn exp(input: Vec<TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap();//c'è solo un input
    // Converti il TensorProto di input in un ndarray di tipo f32.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Ottieni la dimensione del batch dal tensore di input.
    let batch_size = input_nd_array.shape()[0];

    // Applica l'esponenziale a ciascun elemento all'interno di ogni batch.
    let results: Vec<_> = (0..batch_size)
        .into_iter()
        .map(|i| {
            let batch_data = input_nd_array.index_axis(Axis(0), i);
            batch_data.mapv(|el| el.exp())
        })
        .collect();

    // Combina i risultati lungo la dimensione del batch.
    let stacked_result = stack_along_batch_dimension(results)?;

    // Converte il risultato in un TensorProto di output e restituisci.
    convert_to_output_tensor(node, stacked_result)
}
