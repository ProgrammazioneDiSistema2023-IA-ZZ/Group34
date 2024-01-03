use crate::{
    onnx::{NodeProto, TensorProto},
    operations::utils::{
        convert_to_output_tensor, extract_attributes, get_int_attribute,
        stack_along_batch_dimension, tensor_proto_to_ndarray,
    }, utils::OnnxError,
};
use ndarray::prelude::*;
// Funzione pubblica per implementare l'operazione di riduzione somma in un grafo ONNX.

pub fn reducesum(input: Vec<TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap(); //c'è solo un input
                                       // Estrai gli attributi dal nodo ONNX.
    let attributes = extract_attributes(&node.attribute)?;

    // Ottieni gli attributi specifici per la riduzione somma.
    let axis = get_int_attribute(&attributes, "axis", Some(-1))?;
    let keepdims = get_int_attribute(&attributes, "keepdims", Some(1))?;
    let noop = get_int_attribute(&attributes, "noop_with_empty_axes", Some(0))?;

    // Converti TensorProto in ndarray.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Ottieni la dimensione del batch.
    let batch_size = input_nd_array.shape()[0];

    // Lista per contenere i risultati della riduzione somma.
    let mut result_list = Vec::with_capacity(batch_size);

    // Itera su ciascun batch e calcola la riduzione somma.
    for b in 0..batch_size {
        // Estrai il campione dal batch.
        let sample = input_nd_array.index_axis(Axis(0), b);

        // Calcola la riduzione somma in base all'asse specificato.
        let result = if axis == -1 {
            let sum = sample.sum();
            let sum_array: Array<f32, _> = Array::from_elem(IxDyn(&[1]), sum);
            sum_array.into_dyn()
        } else {
            let axis = if axis >= 0 {
                axis as usize
            } else {
                (axis + sample.ndim() as i64) as usize
            };

            // Esegue la riduzione somma sull'asse specificato.
            let reduced_array = sample.sum_axis(Axis(axis));

            // Se keepdims è 0, riduci le dimensioni dell'asse ridotto a 1.
            if keepdims == 0 {
                let mut new_shape = reduced_array.shape().to_vec();
                new_shape[axis] = 1;
                reduced_array.into_shape(new_shape).unwrap()
            } else {
                reduced_array
            }
        };

        // Aggiungi il risultato alla lista.
        result_list.push(result);
    }

    // Unisci i risultati lungo la dimensione del batch.
    let result = stack_along_batch_dimension(result_list)?;

    // Se noop è 0 e la lunghezza del risultato è 0, restituisci il vettore di input originale.
    let result = if noop == 0 && result.len() == 0 {
        input_nd_array.clone()
    } else {
        result
    };

    // Converti il risultato finale in TensorProto e restituisci.
    convert_to_output_tensor(node, result)
}
