use crate::{operations::utils::{
    convert_to_output_tensor, extract_attributes, get_float_attribute, get_int_attribute,
    tensor_proto_to_ndarray,
}, onnx::{TensorProto, NodeProto}, utils::OnnxError};
use ndarray::prelude::*;
#[allow(unused_imports)]
use rayon::prelude::*;
// Funzione pubblica per implementare l'operazione di "Local Response Normalization" (LRN) in un grafo ONNX.

pub fn lrn(input: Vec<TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap();//c'è solo un input
    // Estrai gli attributi dal nodo ONNX.
    let attributes = extract_attributes(&node.attribute)?;
    let alpha: f32 = get_float_attribute(&attributes, "alpha", Some(0.0001))?;
    let beta: f32 = get_float_attribute(&attributes, "beta", Some(0.75))?;
    let bias: f32 = get_float_attribute(&attributes, "bias", Some(1.0))?;
    let size: usize = get_int_attribute(&attributes, "size", None)? as usize;

    // Converti TensorProto in ndarray.
    let x = tensor_proto_to_ndarray::<f32>(input)?;
    let shape = x.dim();
    let c = shape[1]; // Numero di canali

    // Inizializza un array con zeri della stessa forma di x per memorizzare la somma dei quadrati.
    let mut square_sum = Array::zeros(shape);

    // Dividi square_sum in n fette lungo la dimensione del batch e itera in parallelo.
    square_sum
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(b, mut batch_slice)| {
            for idx in 0..c {
                // Calcola gli indici di inizio e fine per la finestra di normalizzazione.
                let start = usize::max(
                    0,
                    idx.saturating_sub(((size - 1) as f32 / 2.0).floor() as usize),
                );
                let end = usize::min(c, idx + ((size - 1) as f32 / 2.0).ceil() as usize);

                // Esegui la somma dei quadrati sulla finestra di normalizzazione.
                for j in start..end {
                    let slice = x.slice(s![b, j, .., ..]);
                    batch_slice
                        .slice_mut(s![idx, .., ..])
                        .zip_mut_with(&slice.mapv(|v| v.powi(2)), |a, &b| *a += b);
                }
            }
        });

    // Calcola il risultato finale applicando la normalizzazione.
    let y = &x / (bias + alpha / (size as f32) * &square_sum).mapv(|v| v.powf(beta));

    // Converti il risultato finale in TensorProto e restituisci.
    convert_to_output_tensor(node, y)
}
