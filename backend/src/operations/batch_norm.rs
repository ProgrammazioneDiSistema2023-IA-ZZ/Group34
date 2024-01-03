use crate::{
    onnx::{NodeProto, TensorProto},
    operations::utils::{
        convert_to_output_tensor, extract_attributes, get_float_attribute,
        stack_along_batch_dimension, tensor_proto_to_ndarray
    }, utils::OnnxError,
};
use ndarray::prelude::*;
use rayon::prelude::*;
pub fn batch_norm(
    input: Vec<TensorProto>,
    initializers: Vec<TensorProto>,
    node: &NodeProto,
    is_par_enabled:  bool,
) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap(); //c'è solo un input
                                       // Estrai gli attributi del nodo.
    let attributes = extract_attributes(&node.attribute)?;
    let epsilon = get_float_attribute(&attributes, "epsilon", Some(1e-05))?;

    // Converti TensorProto in ndarrays.
    let x = tensor_proto_to_ndarray::<f32>(input)?;
    let scale = tensor_proto_to_ndarray::<f32>(&initializers[0])?;
    let bias = tensor_proto_to_ndarray::<f32>(&initializers[1])?;
    let mean = tensor_proto_to_ndarray::<f32>(&initializers[2])?;
    let var = tensor_proto_to_ndarray::<f32>(&initializers[3])?;

    let batch_size = x.shape()[0];

    // Effettua il broadcasting di mean, variance, scale e bias.
    let broadcasted_mean = mean
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast mean".into()))?;

    let broadcasted_var = var
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast variance".into()))?;

    let broadcasted_scale = scale
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast scale".into()))?;

    let broadcasted_bias = bias
        .into_shape((x.shape()[1], 1, 1))
        .map_err(|_| OnnxError::ShapeMismatch("Failed to broadcast bias".into()))?;

    // Calcola la normalizzazione batch per ogni batch.
    if is_par_enabled {
        let result_list : Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let batch_data = x.index_axis(Axis(0), i);

                // Normalizza il batch.
                let normalized =
                    (&batch_data - &broadcasted_mean) / (&broadcasted_var + epsilon).mapv(|v| v.sqrt());

                // Applica scale e bias.
                normalized * &broadcasted_scale + &broadcasted_bias
            })
            .collect();
            // Combina i risultati lungo la dimensione del batch.
            let result = stack_along_batch_dimension(result_list)?;

            // Converte il risultato in formato TensorProto.
            convert_to_output_tensor(node, result)
    }else{
        let result_list: Vec<_> = (0..batch_size)
        .into_iter()
        .map(|i| {
            let batch_data = x.index_axis(Axis(0), i);

            // Normalizza il batch.
            let normalized =
                (&batch_data - &broadcasted_mean) / (&broadcasted_var + epsilon).mapv(|v| v.sqrt());

            // Applica scale e bias.
            normalized * &broadcasted_scale + &broadcasted_bias
        })
        .collect();
        // Combina i risultati lungo la dimensione del batch.
        let result = stack_along_batch_dimension(result_list)?;

        // Converte il risultato in formato TensorProto.
        convert_to_output_tensor(node, result)
    }
}
