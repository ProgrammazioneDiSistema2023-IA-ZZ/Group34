use crate::{operations::utils::{
    convert_to_output_tensor, extract_attributes, get_int_attribute, tensor_proto_to_ndarray,
}, onnx::{NodeProto, TensorProto}, utils::OnnxError};
use ndarray::prelude::*;
use rand::{Rng, SeedableRng};
// Funzione pubblica per l'implementazione del dropout in un grafo ONNX.
// Il dropout è una tecnica di regolarizzazione utilizzata durante l'addestramento dei modelli di apprendimento automatico.
#[allow(unreachable_patterns)]
pub fn dropout(
    input: Vec<TensorProto>,
    initializers: Vec<TensorProto>,
    node: &NodeProto,
) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap();//c'è solo un input
    // Converti il TensorProto di input in un ndarray di tipo f32.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Estrai il tasso di dropout e la modalità di addestramento dagli initializers o imposta i valori predefiniti.
    let (ratio, training_mode) = match initializers {
        tensor_protos => {
            let ratio = tensor_protos
                .get(0)
                .and_then(|tp| tp.float_data.get(0))
                .cloned()
                .unwrap_or(0.5);
            let training_mode = tensor_protos
                .get(1)
                .and_then(|tp| tp.int64_data.get(0))
                .cloned()
                .unwrap_or(0);

            (ratio, training_mode)
        }
        _ => (0.5, 0),
    };

    // Se la modalità di addestramento è 0, restituisci direttamente il risultato senza dropout.
    if training_mode == 0 {
        let result = convert_to_output_tensor(node, input_nd_array.clone());
        return result;
    }

    // Estrai gli attributi dal nodo ONNX.
    let attributes = extract_attributes(&node.attribute)?;

    // Ottieni il seme per il generatore di numeri casuali o utilizza un valore casuale.
    let seed = get_int_attribute(&attributes, "seed", Some(rand::thread_rng().gen()))?;

    // Calcola la scala per compensare il dropout durante l'addestramento.
    let scale = 1. / (1. - ratio);

    // Ottieni la forma dell'input.
    let shape = input_nd_array.shape();

    // Ottieni la forma delle feature, escludendo la dimensione del batch.
    let feature_shape: Vec<_> = shape[1..].to_vec();

    // Inizializza un generatore di numeri casuali con il seme fornito.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);

    // Calcola la lunghezza della maschera di dropout.
    let mask_len = feature_shape.iter().product::<usize>();

    // Genera una maschera di dropout booleana utilizzando il generatore di numeri casuali.
    let single_mask: ArrayD<bool> =
        Array::from_iter(std::iter::repeat_with(|| rng.gen::<f32>() >= ratio).take(mask_len))
            .into_shape(feature_shape)
            .unwrap();

    // Trasmetti la maschera a forma di batch e applica la scala e la maschera all'input.
    let mask = single_mask.broadcast(shape.to_vec()).unwrap();
    let result = input_nd_array.mapv(|x| x * scale) * mask.mapv(|x| if x { 1.0 } else { 0.0 });

    // Converte il risultato in un TensorProto di output e restituisci.
    convert_to_output_tensor(node, result)
}
