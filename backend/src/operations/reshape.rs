use crate::{operations::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, tensor_proto_to_ndarray,
}, onnx::{TensorProto, NodeProto}, utils::OnnxError};
use ndarray::prelude::*;

// Funzione privata per la riduzione di un singolo tensore secondo una forma target,
// consentendo un'eventuale dimensione inferita.

fn reshape_single_tensor(
    input: ArrayViewD<f32>,
    shape: &Vec<isize>,
    allow_zero: i64,
) -> Result<Array<f32, IxDyn>, OnnxError> {
    // Variabile per tenere traccia di una possibile dimensione inferita.
    let mut inferred_dim = None;
    let mut target_shape = shape.clone();

    // Itera sulla forma target per risolvere eventuali dimensioni inferite o zero.
    for (i, dim) in target_shape.iter_mut().enumerate() {
        if *dim == -1 {
            if inferred_dim.is_some() {
                return Err(OnnxError::ShapeMismatch(
                    "More than one inferred dimension!".into(),
                ));
            }
            inferred_dim = Some(i);
        } else if *dim == 0 {
            if allow_zero == 0 {
                *dim = input.shape()[i] as isize;
            }
        }
    }

    // Se presente una dimensione inferita, calcola la sua dimensione.
    if let Some(idx) = inferred_dim {
        let product_of_dims: isize = target_shape.iter().filter(|&&dim| dim != -1).product();
        target_shape[idx] = (input.len() as isize) / product_of_dims;
    }

    // Applica la nuova forma al tensore.
    Ok(input
        .into_shape(target_shape.iter().map(|&x| x as usize).collect::<Vec<_>>())
        .unwrap()
        .to_owned())
}

// Funzione privata per la riduzione di un tensore con operazioni batch.

fn reshape_with_batches(
    input: &TensorProto,
    parameter: &TensorProto,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Estrai attributi specifici del nodo.
    let attributes = extract_attributes(&node.attribute)?;
    let allow_zero: i64 = get_int_attribute(&attributes, "allow_zero", Some(0))?;

    // Recupera il tensore dati sia da `inputs` che da `initializers`.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(&input)?;

    // Determina il tensore di forma.
    let shape_tensor = parameter;

    let mut target_shape: Vec<isize> = tensor_proto_to_ndarray::<i64>(&shape_tensor)?
        .into_raw_vec()
        .iter()
        .map(|&x| x as isize)
        .collect();

    target_shape[0] *= input_nd_array.shape()[0] as isize;

    // Riduci ogni batch separatamente.
    let reshaped_batches =
        reshape_single_tensor(input_nd_array.view(), &target_shape, allow_zero).unwrap();

    convert_to_output_tensor(node, reshaped_batches)
}

// Funzione privata per la riduzione di un tensore senza operazioni batch.

fn reshape_without_batches(
    initializers: Vec<TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    // Estrai attributi specifici del nodo.
    let attributes = extract_attributes(&node.attribute)?;
    let allow_zero: i64 = get_int_attribute(&attributes, "allow_zero", Some(0))?;

    let input_nd_array = tensor_proto_to_ndarray::<f32>(&initializers[0])?;
    let target_shape: Vec<isize> = tensor_proto_to_ndarray::<i64>(&initializers[1])?
        .into_raw_vec()
        .iter()
        .map(|&x| x as isize)
        .collect();

    // Riduci il tensore.
    let reshaped = reshape_single_tensor(input_nd_array.view(), &target_shape, allow_zero)?;

    convert_to_output_tensor(node, reshaped)
}

//I principi fondamentali del processo di riduzione includono:
// 1. Solo una dimensione nella nuova forma può avere un valore di -1. In questo scenario, il valore della
//    dimensione è dedotto dalle dimensioni del tensore e da eventuali dimensioni residue.
// 2. Una dimensione potrebbe essere assegnata un valore di 0. Se l'attributo `allowzero` non è impostato,
//    il valore originale della dimensione rimane invariato (estratto dal tensore di input). Se `allowzero`
//    è attivo e la nuova forma contiene uno 0, questa dimensione verrà esplicitamente impostata a zero.
// 3. Il numero totale di elementi sia nella forma del tensore di input che nella forma del tensore di output
//    deve essere identico.
//
// Si noti che specificare una forma che include sia uno 0 che un valore di -1 è invalido quando l'attributo
// `allowzero` è attivato.
pub fn reshape(
    inputs: Vec<TensorProto>,
    initializers: Vec<TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    if initializers.len() == 2 {
        reshape_without_batches(initializers, node)
    } else {
        reshape_with_batches(&inputs[0], &initializers[0], node)
    }
}
