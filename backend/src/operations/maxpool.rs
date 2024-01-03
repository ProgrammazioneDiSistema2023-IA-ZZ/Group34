use crate::{operations::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, get_ints_attribute,
    pad_matrix_2d, tensor_proto_to_ndarray,
}, onnx::{TensorProto, NodeProto}, utils::OnnxError};
use ndarray::prelude::*;
use rayon::prelude::*;
// Funzione pubblica per implementare l'operazione di max pooling in un grafo ONNX.

pub fn maxpool(inputs: Vec<TensorProto>, node: &NodeProto,is_par_enabled:  bool,) -> Result<TensorProto, OnnxError> {
    let inputs = inputs.get(0).unwrap();//c'è solo un input
    // Estrai gli attributi dal nodo ONNX.
    let attributes = extract_attributes(&node.attribute)?;

    // Le dimensioni del kernel e dello stride del max pooling sono sempre bidimensionali.
    let kernel_shape = get_ints_attribute(&attributes, "kernel_shape", Some(vec![1, 1]))?;
    let pads = get_ints_attribute(&attributes, "pads", Some(vec![0, 0, 0, 0]))?;
    let strides = get_ints_attribute(&attributes, "strides", Some(vec![1, 1]))?;

    // TODO: ceil_mode, storage_order e dilations non sono utilizzati
    let _ceil_mode = get_int_attribute(&attributes, "ceil_mode", Some(0))?;
    let _storage_order = get_int_attribute(&attributes, "storage_order", Some(0))?;
    let _dilations = get_ints_attribute(&attributes, "dilations", Some(vec![1, 1]))?;

    // Converti TensorProto in ndarray.
    let inputs_nd_array = tensor_proto_to_ndarray::<f32>(&inputs)?;

    // Calcola il risultato del max pooling.
    let result = pool(&inputs_nd_array, &kernel_shape, &pads, &strides,is_par_enabled)?;

    // Converti il risultato finale in TensorProto e restituisci.
    convert_to_output_tensor(node, result)
}

// Funzione di supporto per implementare l'operazione di max pooling.
fn pool(
    input_matrix: &ArrayD<f32>,
    kernel_shape: &Vec<i64>,
    pads: &Vec<i64>,
    strides: &Vec<i64>,
    is_par_enabled: bool,
) -> Result<ArrayD<f32>, OnnxError> {
    // Scegli gli indici delle dimensioni da estrarre.
    let batch_size = input_matrix.shape()[0];
    let channels = input_matrix.shape()[1];

    // Estrai la forma del kernel e degli stride.
    let kernel_height = kernel_shape[0] as usize;
    let kernel_width = kernel_shape[1] as usize;
    let stride_height = strides[0] as usize;
    let stride_width = strides[1] as usize;

    let mut pooled_results = Vec::new();

    // Itera su batch e canali per eseguire il max pooling.
    for b in 0..batch_size {
        for c in 0..channels {
            // Estrai la matrice di input bidimensionale (altezza x larghezza).
            let h_w_matrix = input_matrix
                .index_axis(Axis(0), b)
                .index_axis(Axis(0), c)
                .to_owned()
                .into_shape((input_matrix.shape()[2], input_matrix.shape()[3]))
                .unwrap();

            // Applica il padding alla matrice di input bidimensionale.
            let padded_matrix = pad_matrix_2d(&h_w_matrix, &pads)?;

            // Estrai le dimensioni della matrice con padding.
            let (padded_rows, padded_cols) = padded_matrix.dim();

            // Calcola le dimensioni dell'output.
            let output_rows = (padded_rows - kernel_height) / stride_height + 1;
            let output_cols = (padded_cols - kernel_width) / stride_width + 1;

            // Parallelizza l'operazione di max pooling
            if is_par_enabled==true{
                let pooled_matrix = (0..output_rows)
                    .into_par_iter()
                    .map(|i| {
                        let mut row = Vec::with_capacity(output_cols);

                        for j in 0..output_cols {
                            let start_row = i * stride_height;
                            let start_col = j * stride_width;
                            let end_row = start_row + kernel_height;
                            let end_col = start_col + kernel_width;

                            // Estrai la patch corrispondente dalla matrice con padding.
                            let patch = padded_matrix.slice(s![start_row..end_row, start_col..end_col]);

                            // Calcola il valore massimo all'interno della patch.
                            let max_value = patch.fold(std::f32::NEG_INFINITY, |acc, &x| acc.max(x));

                            row.push(max_value);
                        }

                        row
                    })
                    .collect::<Vec<_>>();

                pooled_results.push(pooled_matrix);
            }else{
                    let pooled_matrix = (0..output_rows)
                    .into_iter()
                    .map(|i| {
                        let mut row = Vec::with_capacity(output_cols);

                        for j in 0..output_cols {
                            let start_row = i * stride_height;
                            let start_col = j * stride_width;
                            let end_row = start_row + kernel_height;
                            let end_col = start_col + kernel_width;

                            // Estrai la patch corrispondente dalla matrice con padding.
                            let patch = padded_matrix.slice(s![start_row..end_row, start_col..end_col]);

                            // Calcola il valore massimo all'interno della patch.
                            let max_value = patch.fold(std::f32::NEG_INFINITY, |acc, &x| acc.max(x));

                            row.push(max_value);
                        }

                        row
                    })
                    .collect::<Vec<_>>();

                pooled_results.push(pooled_matrix);
            }
        }
    }

    // Converte le righe raccolte in un ArrayD.
    let pooled_tensor = ArrayD::from_shape_vec(
        IxDyn(&[
            batch_size,
            channels,
            pooled_results[0].len(),
            pooled_results[0][0].len(),
        ]),
        pooled_results.into_iter().flatten().flatten().collect(),
    )
    .map_err(|_| OnnxError::ShapeError("Failed to create output tensor".to_string()));

    pooled_tensor
}
