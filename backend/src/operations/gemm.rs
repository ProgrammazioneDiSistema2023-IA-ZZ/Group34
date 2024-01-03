use crate::{
    onnx::{NodeProto, TensorProto},
    operations::utils::{
        convert_to_output_tensor, extract_attributes, get_float_attribute, get_int_attribute,
        tensor_proto_to_ndarray,
    }, utils::OnnxError,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use tract_onnx::tract_core::tract_data::itertools::Itertools;
pub enum OperationMode {
    Gemm,
    Matmul,
}

// Funzione pubblica per implementare l'operazione di moltiplicazione di matrici o gemm in un grafo ONNX.

pub fn gemm(
    inputs: Vec<TensorProto>,
    initializers: Vec<TensorProto>,
    node: &NodeProto,
    is_par_enabled: bool,
) -> Result<TensorProto, OnnxError> {
    // Estrai gli attributi dal nodo ONNX.
    let attributes = extract_attributes(&node.attribute)?;

    // Determina la modalità di operazione (Gemm o MatMul) dal tipo di operazione nel nodo.
    let mode = determine_mode(&node.op_type)?;

    // Ottieni i valori degli attributi alpha, beta, transA e transB dal nodo.
    let alpha: f32 = get_float_attribute(&attributes, "alpha", Some(1.0))?;
    let beta: f32 = get_float_attribute(&attributes, "beta", Some(0.0))?;
    let trans_a: i64 = get_int_attribute(&attributes, "transA", Some(0))?;
    let trans_b: i64 = get_int_attribute(&attributes, "transB", Some(0))?;

    // Unisci i TensorProto di input e i TensorProto inizializzatori.
    let mut merged_tensors: Vec<TensorProto> = inputs.clone();
    merged_tensors.extend(initializers);

    // Converti i TensorProto A e B in ndarray di tipo f32.
    let mut a =
        tensor_proto_to_ndarray::<f32>(get_tensor(&merged_tensors.iter().collect_vec(), 0, "A")?)?;
    let mut b =
        tensor_proto_to_ndarray::<f32>(get_tensor(&merged_tensors.iter().collect_vec(), 1, "B")?)?;

    // Trasponi le matrici A e B se richiesto dagli attributi transA e transB.
    if trans_a == 1 {
        a = a.t().to_owned();
    }
    if trans_b == 1 {
        b = b.t().to_owned();
    }

    // Esegui la moltiplicazione delle matrici.
    let mut result = matrix_multiply(&a, &b, is_par_enabled).ok_or(OnnxError::InternalError(
        "Failed to multiply matrices".to_string(),
    ))?;

    // Moltiplica il risultato per il valore alpha.
    result.mapv_inplace(|x| x * alpha);

    // Se la modalità è Gemm, applica la moltiplicazione per il valore beta e somma il tensor C.
    if let OperationMode::Gemm = mode {
        if let Some(c_tensor_proto) = inputs.get(2) {
            let mut c_array = tensor_proto_to_ndarray::<f32>(c_tensor_proto)?;
            c_array.mapv_inplace(|x| x * beta);

            // Verifica la corrispondenza delle forme tra il risultato e il tensor C.
            if result.shape() != c_array.shape() {
                return Err(OnnxError::ShapeMismatch(format!(
                    "Expected shape {:?}, but got {:?}",
                    result.shape(),
                    c_array.shape()
                )));
            }

            // Somma il tensor C al risultato.
            result += &c_array;
        }
    }

    // Converte il risultato in un TensorProto di output e restituisci.
    convert_to_output_tensor(node, result)
}

// Funzione di supporto per determinare la modalità di operazione (Gemm o MatMul) dal tipo di operazione.
fn determine_mode(op_type: &str) -> Result<OperationMode, OnnxError> {
    match op_type {
        "Gemm" => Ok(OperationMode::Gemm),
        "MatMul" => Ok(OperationMode::Matmul),
        _ => Err(OnnxError::InternalError(format!(
            "Unsupported operation: {}",
            op_type
        ))),
    }
}

// Funzione di supporto per ottenere il TensorProto da un vettore di input.
fn get_tensor<'a>(
    inputs: &'a Vec<&'a TensorProto>,
    index: usize,
    name: &str,
) -> Result<&'a TensorProto, OnnxError> {
    inputs
        .get(index)
        .copied()
        .ok_or(OnnxError::MissingInput(name.to_string()))
}

// Funzione di supporto per eseguire la moltiplicazione di matrici in batch o singola in base alle dimensioni di A.
fn matrix_multiply(a: &ArrayD<f32>, b: &ArrayD<f32>, is_par_enabled: bool) -> Option<ArrayD<f32>> {
    let b_matrix = b
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .to_owned();

    match a.shape()[0] {
        1 => matrix_multiply_single(a, &b_matrix),
        _ => matrix_multiply_batched(a, &b_matrix, is_par_enabled),
    }
}

// Funzioni di supporto per eseguire la moltiplic

fn matrix_multiply_batched(
    a: &ArrayD<f32>,
    b_matrix: &ndarray::Array2<f32>,
    is_par_enabled: bool,
) -> Option<ArrayD<f32>> {
    let shape = a.shape();
    let batch_size = shape[0];

    // Parallel processing of the batch
    if is_par_enabled {
        let result_list: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let a_slice = a.slice(s![i, ..]);
                a_slice.dot(b_matrix)
            })
            .collect();

        let views: Vec<_> = result_list.iter().map(|arr| arr.view()).collect();
        let result = ndarray::stack(Axis(0), &views[..]).unwrap();

        Some(result.into_dyn())
    } else {
        let result_list: Vec<_> = (0..batch_size)
            .into_iter()
            .map(|i| {
                let a_slice = a.slice(s![i, ..]);
                a_slice.dot(b_matrix)
            })
            .collect();

        let views: Vec<_> = result_list.iter().map(|arr| arr.view()).collect();
        let result = ndarray::stack(Axis(0), &views[..]).unwrap();

        Some(result.into_dyn())
    }
}

fn matrix_multiply_single(a: &ArrayD<f32>, b_matrix: &ndarray::Array2<f32>) -> Option<ArrayD<f32>> {
    let a_2d = a.to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let result = a_2d.dot(b_matrix);
    Some(result.into_dyn())
}
