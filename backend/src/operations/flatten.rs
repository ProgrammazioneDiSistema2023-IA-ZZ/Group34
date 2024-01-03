use crate::{operations::utils::{convert_to_output_tensor, tensor_proto_to_ndarray}, onnx::{TensorProto, NodeProto}, utils::OnnxError};
// Funzione pubblica per implementare l'operazione di flattening in un grafo ONNX.
// L'operazione di flattening ridimensiona il tensore di input a una forma lineare (1D).

pub fn flatten(input: Vec<TensorProto>, node: &NodeProto) -> Result<TensorProto, OnnxError> {
    let input = input.get(0).unwrap();//c'è solo un input
    // Ottieni le dimensioni del tensore di input.
    let input_shape = &input.dims;
    let input_first = input_shape[0] as usize;

    // Converti il TensorProto di input in un ndarray di tipo f32.
    let input_nd_array = tensor_proto_to_ndarray::<f32>(input).map_err(|_| {
        OnnxError::ConversionError("Failed to convert TensorProto to ndarray".into())
    })?;

    // Inizializza la forma del risultato come un vettore di usize vuoto.
    let mut output_shape: Vec<usize> = Vec::new();

    // Ottieni l'attributo "axis" dal nodo ONNX.
    let axis_attribute = &node
        .attribute
        .iter()
        .find(|attr| attr.name == "axis");

    // Se l'attributo "axis" non è presente, assume il valore predefinito 1.
    let axis = axis_attribute.map_or(1, |attr| attr.i as usize);

    // Calcola il numero totale di elementi nel tensore, escludendo la dimensione del batch.
    let total_elements = input_shape.clone()[1..].iter().product::<i64>() as usize;

    // Determina la forma del risultato in base all'asse di flattening.
    if axis <= 1 {
        output_shape = vec![input_first, total_elements];
    } else {
        let mut outer_dim = 1;
        let mut inner_dim = 1;

        // Calcola le dimensioni esterne e interne del risultato in base all'asse di flattening.
        for (index, &dim) in input_shape.iter().enumerate() {
            if index < axis {
                outer_dim *= dim as usize;
            } else {
                inner_dim *= dim as usize;
            }
        }

        // Aggiungi le dimensioni calcolate al vettore della forma del risultato.
        output_shape.push(outer_dim);
        output_shape.push(inner_dim);
    }

    // Ridimensiona il tensore di input utilizzando la forma calcolata.
    let result = input_nd_array.into_shape(output_shape).unwrap();

    // Stampa la forma del risultato a scopo di debug.
    println!("shape {:?}", result.shape());

    // Converte il risultato in un TensorProto di output e restituisci.
    convert_to_output_tensor(node, result)
}
