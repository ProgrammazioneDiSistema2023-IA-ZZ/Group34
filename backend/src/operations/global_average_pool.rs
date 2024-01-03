use crate::{
    onnx::{NodeProto, TensorProto},
    operations::utils::{
        convert_to_output_tensor, extract_attributes, stack_along_batch_dimension,
        tensor_proto_to_ndarray,
    }, utils::OnnxError,
};
use ndarray::prelude::*;
use rayon::prelude::*;
// Funzione pubblica per implementare l'operazione di "global average pooling" in un grafo ONNX.

pub fn globalavgpool(
    inputs: Vec<TensorProto>,
    node: &NodeProto,
    is_par_enabled: bool,
) -> Result<TensorProto, OnnxError> {
    let inputs = inputs.get(0).unwrap(); //c'è solo un input
                                         // Estrai gli attributi dal nodo ONNX (non utilizzati in questa funzione, ma potrebbero servire per estensioni future).
    let _attributes = extract_attributes(&node.attribute)?;

    // Converti il TensorProto di input in un ndarray.
    let inputs_nd_array = tensor_proto_to_ndarray::<f32>(inputs)?;

    // Esegui il "global average pooling" sull'ndarray.
    let result = global_average_pooling(&inputs_nd_array, is_par_enabled)?;

    // Converti l'ndarray risultante nuovamente in TensorProto e restituisci.
    convert_to_output_tensor(node, result)
}
fn global_average_pooling(
    input_tensor: &ArrayD<f32>,
    is_par_enabled: bool,
) -> Result<ArrayD<f32>, OnnxError> {
    // Ottieni la dimensione del batch e il numero di canali dalla forma del tensore di input.
    let batch_size = input_tensor.shape()[0];
    let channels = input_tensor.shape()[1];

    // Esegui il "global average pooling" per ogni batch e canale.
    if is_par_enabled {
        let pooled_results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|b| {
                let mut channel_averages = vec![0.0; channels];
                for c in 0..channels {
                    let channel_data = input_tensor
                        .index_axis(Axis(0), b)
                        .index_axis(Axis(0), c)
                        .to_owned();

                    // Calcola il valore medio per il canale corrente.
                    let average_value = channel_data.mean().unwrap();
                    channel_averages[c] = average_value;
                }
                // Converti le medie dei canali in un tensore di forma [channels, 1, 1].
                ArrayD::from_shape_vec(IxDyn(&[channels, 1, 1]), channel_averages)
                    .expect("Failed to create tensor from averages")
            })
            .collect();

        // Impila i risultati lungo la dimensione del batch per produrre il tensore di output finale.
        stack_along_batch_dimension(pooled_results)
    } else {
        let pooled_results: Vec<_> = (0..batch_size)
            .into_iter()
            .map(|b| {
                let mut channel_averages = vec![0.0; channels];
                for c in 0..channels {
                    let channel_data = input_tensor
                        .index_axis(Axis(0), b)
                        .index_axis(Axis(0), c)
                        .to_owned();

                    // Calcola il valore medio per il canale corrente.
                    let average_value = channel_data.mean().unwrap();
                    channel_averages[c] = average_value;
                }
                // Converti le medie dei canali in un tensore di forma [channels, 1, 1].
                ArrayD::from_shape_vec(IxDyn(&[channels, 1, 1]), channel_averages)
                    .expect("Failed to create tensor from averages")
            })
            .collect();

        // Impila i risultati lungo la dimensione del batch per produrre il tensore di output finale.
        stack_along_batch_dimension(pooled_results)
    }
}
