mod onnx {
    include!("onnx.rs");
}

use std::io::ErrorKind;
use ndarray::{arr2, Array, Array2, Array4, ArrayD, ArrayView};
use onnx::tensor_shape_proto::Dimension;
use protoc::Error;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::tract_itertools::Itertools;
use onnx::ModelProto;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;
use rand::prelude::*;

fn main() {
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    let parsed_proto: ModelProto = prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    // Use the parsed ProtoBuf data as needed
    let i = parsed_proto.graph.unwrap().initializer.get(10).unwrap().clone();
    let j = i.clone();
    let matrix = Array4::from_shape_vec((64, 16,1,1), i.float_data).unwrap();
    println!("{:?}", op_add(vec![i, j], vec![]));


}

struct Operation{
    op_type: OperationType,
    input: Vec<TensorProto>,
    op_attributes: Vec<AttributeProto>
}

enum OperationType{
    ADD,
    RELU,
    EXP,
    CONCAT,
    FLATTEN,
    RESHAPE,
    CONV,
    MAXPOOL,
    BATCHNORM,
    DROPOUT,
    SOFTMAX,
    GEMM,
    MATMUL,
    REDUCESUM,
    GLOBALAVGPOOL,
    LRN

}

fn perform_operation(op: Operation) -> Result<TensorProto, Error>{
    match op.op_type {
        OperationType::ADD => {
            op_add(op.input, op.op_attributes)
        }
        OperationType::RELU => {
            op_relu(op.input)
        }
        OperationType::EXP => {
            op_exp
        }
        OperationType::CONCAT => {
            return
        }
        OperationType::FLATTEN => {
            return
        }
        OperationType::RESHAPE => {
            return ;
        }
        OperationType::CONV => {
            return
        }
        OperationType::MAXPOOL => {
            return
        }
        OperationType::BATCHNORM => {
            return
        }
        OperationType::DROPOUT => {
            return
        }
        OperationType::SOFTMAX => {
            return
        }
        OperationType::GEMM => {
            return
        }
        OperationType::MATMUL => {
            return
        }
        OperationType::REDUCESUM => {
            return
        }
        OperationType::GLOBALAVGPOOL => {
            return 
        }
        OperationType::LRN => {
            return 
        }
        _ => {
            Err(Error::new(ErrorKind::Other, "Incorrect operation!"))
        }
    }
}

fn op_add(input: Vec<TensorProto>, op_attributes: Vec<AttributeProto>) -> Result<TensorProto, Error>{
    let a: ArrayD<f32> = input.get(0).unwrap().into();
    let b: ArrayD<f32> = input.get(0).unwrap().into();
    let c = a+b;
    println!("{:?}", c);
    return Ok(TensorProto::from(c));
    
}

fn op_relu(input: Vec<TensorProto>)->Result<TensorProto, Error>{
    let a: ArrayD<f32> = input.get(0).unwrap().into();
    for i in 0..a.len() {
        if a[i].clone() < 0.0 {
            a[i] = 0.0;
        }
    }
    return Ok(TensorProto::from(a));
}
fn op_exp(){

}
fn op_concat<'a, A, D>(arrays: &[ArrayView<'a, A, D>], axis: i32) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
    {
    /*
        In ONNX (Open Neural Network Exchange), l'operazione Concat (Concatenazione) è utilizzata per concatenare più tensori lungo una dimensione specifica. Questa operazione è comunemente utilizzata nelle reti neurali per combinare le informazioni provenienti da diverse parti della rete.

    La firma dell'operazione Concat in ONNX è la seguente:

    proto
    Copy code
    node {
    input: "input_1"
    input: "input_2"
    output: "output"
    op_type: "Concat"
    attribute {
        name: "axis"
        i: 1  # L'asse lungo il quale eseguire la concatenazione
    }
    }
    Dove:

    input_1 e input_2 sono gli input che devono essere concatenati.
    output è il risultato della concatenazione.
    axis specifica l'asse lungo il quale eseguire la concatenazione. Ad esempio, se axis è impostato su 1, i tensori verranno concatenati lungo la dimensione delle colonne.
    Esempio:
    Supponiamo di avere due tensori:

    Tensor1:

    lua
    Copy code
    [[1, 2],
    [3, 4]]
    Tensor2:

    lua
    Copy code
    [[5, 6],
    [7, 8]]
    Se eseguiamo una concatenazione lungo l'asse 1, otteniamo:

    lua
    Copy code
    [[1, 2, 5, 6],
    [3, 4, 7, 8]]
     */
    // Verifica che le dimensioni siano compatibili
    let concat_size: usize = arrays.iter().map(|a| a.shape()[axis]).sum();
    let mut concat_shape = arrays[0].shape().to_vec();
    concat_shape[axis.index()] = concat_size;

    // Creazione dell'array risultante
    let mut result = Array::zeros(concat_shape.clone());

    // Copia dei dati negli array risultanti
    let mut offset = 0;
    for array in arrays {
        let mut slices = result.slice_mut(s![..;, ..;]);
        slices.index_axis_mut(axis, offset..offset + array.shape()[axis.index()]).assign(array);
        offset += array.shape()[axis.index()];
    }

    result


}
fn op_flatten(){

}
fn op_reshape<A: Clone, D: ndarray::Dimension>(input: &Array<A, D>, new_shape: D) -> Array<A, D> {
    input.clone().into_shape(new_shape).unwrap()

    /*
        L'operazione di Reshape in ONNX (Open Neural Network Exchange) è utilizzata per cambiare la forma (dimensione) di un tensore senza modificarne i dati. Ciò può essere utile, ad esempio, quando si vogliono adattare i risultati di un layer per essere compatibili con il layer successivo.

    Ecco come potrebbe apparire un nodo ONNX per l'operazione di Reshape:

    proto
    Copy code
    node {
    input: "input"
    output: "output"
    op_type: "Reshape"
    attribute {
        name: "shape"
        ints: 2  # Nuove dimensioni desiderate
        ints: 3
    }
    }
    Dove:

    input è il tensore che si desidera ridimensionare.
    output è il tensore risultante dopo la modifica della forma.
    op_type è "Reshape".
    shape specifica le nuove dimensioni desiderate per il tensore risultante.
    Ad esempio, se hai un tensore 2D 
    4
    ×
    2
    4×2, puoi utilizzare l'operazione di Reshape per trasformarlo in un tensore 2D 
    2
    ×
    4
    2×4.

    Nel tuo caso, con l'operazione di Reshape con le nuove dimensioni (2, 3), stai cambiando la forma del tensore in modo che abbia due dimensioni, con dimensioni rispettivamente pari a 2 e 3.

    Puoi eseguire l'operazione di Reshape anche con i framework di deep learning come TensorFlow o PyTorch. L'implementazione specifica varierà a seconda del framework che stai utilizzando. 
     */
}
fn op_conv<A, D>(input: &Array<A, D>, kernel: &Array<A, D>) -> Array<A, D>
where
    A: Clone + std::ops::Mul<Output = A> + std::ops::Add<Output = A> + Default,
    D: Dimension,
{
    let (input_rows, input_cols) = input.dim();
    let (kernel_rows, kernel_cols) = kernel.dim();

    let mut output = Array::default(input.dim());

    for i in 0..(input_rows - kernel_rows + 1) {
        for j in 0..(input_cols - kernel_cols + 1) {
            let input_patch = input.slice(s![i..(i + kernel_rows), j..(j + kernel_cols)]);
            let convolution_result = &input_patch * kernel;
            output.slice_mut(s![i, j]).assign(&convolution_result.sum());
        }
    }

    output
    /*
            L'operazione di convoluzione in ONNX (Open Neural Network Exchange) è utilizzata per eseguire la convoluzione di un tensore di input con un kernel. Questa operazione è comune nelle reti neurali convoluzionali (CNN) per l'estrazione di features da immagini o altri dati con struttura simile.

    Ecco come potrebbe apparire un nodo ONNX per l'operazione di convoluzione:

    proto
    Copy code
    node {
    input: "input"   # Tensore di input
    input: "kernel"  # Kernel di convoluzione
    output: "output" # Tensore risultante dopo la convoluzione
    op_type: "Conv"  # Tipo di operazione: convoluzione
    attribute {
        name: "strides"
        ints: 1  # Passo di spostamento sull'asse delle righe
        ints: 1  # Passo di spostamento sull'asse delle colonne
    }
    attribute {
        name: "pads"
        ints: 0  # Padding superiore
        ints: 0  # Padding sinistro
        ints: 0  # Padding inferiore
        ints: 0  # Padding destro
    }
    }
    Dove:

    input è il tensore di input.
    kernel è il kernel di convoluzione.
    output è il tensore risultante dopo la convoluzione.
    op_type è "Conv", indicando che si tratta di un'operazione di convoluzione.
    strides specifica il passo di spostamento lungo le dimensioni dell'input durante la convoluzione.
    pads specifica il padding applicato alle estremità dell'input.
    Nel tuo modello ONNX, dovresti specificare i nomi dei tensori di input, output e kernel in base alla tua architettura specifica.

    Puoi eseguire l'operazione di convoluzione anche con i framework di deep learning come TensorFlow o PyTorch. L'implementazione specifica varierà a seconda del framework che stai utilizzando.
     */
}
fn op_maxpool(){

    /*
    L'operazione di Max Pooling è una tecnica di sottocampionamento utilizzata in reti neurali convoluzionali (CNN) per ridurre la dimensione spaziale di una rappresentazione (immagine, feature map) mantenendo le caratteristiche più importanti. La sua implementazione più comune è il Max Pooling, che opera suddividendo l'input in regioni (o kernel) e restituendo il valore massimo di ciascuna regione.

    Ecco una breve descrizione di come funziona l'operazione di Max Pooling:

    Input:

    L'input è solitamente una feature map o un tensore bidimensionale (ad esempio, un'immagine) con una certa profondità (canali).
    Kernel (Finestra di Max Pooling):

    Si definisce una finestra (o kernel) di dimensioni specifiche (ad esempio, 2x2 o 3x3).
    La finestra scorre sull'input con uno stride specificato (il passo con cui si muove).
    Max Pooling:

    Per ciascuna posizione della finestra, si estraggono i valori corrispondenti dalla feature map.
    Viene restituito il valore massimo tra questi valori (Max Pooling).
    Output:

    Il risultato è una nuova rappresentazione (feature map) con una dimensione ridotta.
    Esempio:
    Supponiamo di avere una matrice di input 4x4:

    lua
    Copy code
    [[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]]
    Con una finestra di Max Pooling 2x2 e uno stride di 2, otteniamo l'output seguente:

    lua
    Copy code
    [[6, 8],
    [14, 16]]
     */
    let (input_rows, input_cols) = matrix.dim();
    let (kernel_rows, kernel_cols) = kernel_shape;
    let output_rows = (input_rows - kernel_rows) / strides.0 + 1;
    let output_cols = (input_cols - kernel_cols) / strides.1 + 1;


    // trovare come avere shapes stripes ecc
    let mut result = Array2::zeros((output_rows, output_cols));

    for i in 0..output_rows {
        for j in 0..output_cols {
            let start_row = i * strides.0;
            let end_row = start_row + kernel_rows;
            let start_col = j * strides.1;
            let end_col = start_col + kernel_cols;

            // Estrai la sottomatrice
            let submatrix = matrix.slice(s![start_row..end_row, start_col..end_col]);

            // Trova il massimo nella sottomatrice e assegnalo alla posizione corrispondente nella matrice risultante
            result[[i, j]] = *submatrix.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        }
    }

    result

}
fn op_batchnorm(){

}
fn op_dropout(input: Vec<TensorProto>, dropout_prob: f64) {
    let mut rng = thread_rng();

    for value in input.get(0).unwrap().iter_mut() {
        let random_value: f64 = rng.gen();  // Genera un numero casuale tra 0.0 e 1.0
        if random_value < dropout_prob {
            *value = 0.0;  // Imposta il valore a zero con probabilità dropout_prob
        }
    }
    /*
        L'operazione Dropout in ONNX (Open Neural Network Exchange) è utilizzata durante l'addestramento delle reti neurali per introdurre casualmente l'eliminazione (dropout) di alcuni neuroni durante la fase di forward pass. Questo aiuta a prevenire l'overfitting, migliorando la generalizzazione del modello.

    Ecco come potrebbe apparire un nodo ONNX per l'operazione di Dropout:

    proto
    Copy code
    node {
    input: "input"
    output: "output"
    op_type: "Dropout"
    attribute {
        name: "ratio"
        f: 0.5  # Probabilità di dropout (0.5 indica il 50% di dropout)
    }
    }
    Dove:

    input è l'input al quale si applica il dropout.
    output è l'output dopo l'applicazione del dropout.
    op_type è "Dropout".
    ratio specifica la probabilità di dropout per ciascun elemento dell'input. Un valore di 0.5 indica che ciascun elemento ha il 50% di probabilità di essere eliminato.
    Puoi utilizzare il Dropout in combinazione con altri strati di una rete neurale per introdurre casualmente dropout durante l'addestramento. Ad esempio, in PyTorch o TensorFlow, solitamente lo strato di Dropout viene aggiunto tra gli strati densi o convoluzionali della rete.

    Per implementare l'operazione di Dropout in un framework di deep learning in Rust, dovresti cercare se la libreria che stai utilizzando supporta l'operazione di Dropout e seguire la documentazione relativa. Attualmente, l'implementazione specifica potrebbe variare a seconda della libreria che stai usando.
     */

}
fn op_softmax(){

}
fn op_gemm(){

}
fn op_matmul(){

}
fn op_reducesum(){

}
fn op_globalavgpool(input: &Array2<f64>, kernel_shape: (usize, usize), pads: (usize, usize, usize, usize), strides: (usize, usize)) -> Array2<f64> {
    let (input_rows, input_cols) = input.dim();
    let (kernel_rows, kernel_cols) = kernel_shape;
    let output_rows = (input_rows - kernel_rows + pads.0 + pads.2) / strides.0 + 1;
    let output_cols = (input_cols - kernel_cols + pads.1 + pads.3) / strides.1 + 1;

    let mut output = Array2::zeros((output_rows, output_cols));

    for i in 0..output_rows {
        for j in 0..output_cols {
            let start_row = i * strides.0;
            let end_row = start_row + kernel_rows;
            let start_col = j * strides.1;
            let end_col = start_col + kernel_cols;

            // Estrai la sottomatrice
            let submatrix = input.slice(Slice::from((start_row..end_row, start_col..end_col)));

            // Calcola la media della sottomatrice e assegnala alla posizione corrispondente nell'output
            output[[i, j]] = submatrix.mean().unwrap();
        }
    }

    output
 /* * 
    L'operazione di Average Pooling in ONNX (Open Neural Network Exchange) è utilizzata per eseguire il sottocampionamento di una feature map calcolando la media dei valori all'interno di una finestra (kernel). Questo aiuta a ridurre le dimensioni della feature map, preservando le caratteristiche più rilevanti.

    Ecco un esempio di come potrebbe apparire un nodo ONNX per Average Pooling:

    proto
    Copy code
    node {
    input: "input"
    output: "output"
    op_type: "AveragePool"
    attribute {
        name: "kernel_shape"
        ints: 3  # Altezza e larghezza del kernel
        ints: 3
    }
    attribute {
        name: "pads"
        ints: 0  # Padding superiore
        ints: 0  # Padding sinistro
        ints: 0  # Padding inferiore
        ints: 0  # Padding destro
    }
    attribute {
        name: "strides"
        ints: 2  # Passo di spostamento sull'asse delle righe
        ints: 2  # Passo di spostamento sull'asse delle colonne
    }
    }
    Dove:

    input è l'input della feature map.
    output è l'output della feature map sottocampionata.
    op_type è "AveragePool".
    kernel_shape specifica le dimensioni della finestra di pooling (ad esempio, 3x3).
    pads specifica il padding applicato alle estremità della feature map (in questo caso, nessun padding).
    strides specifica quanto la finestra di pooling si sposta lungo le dimensioni dell'input.
    Questo esempio rappresenta un nodo Average Pooling che utilizza una finestra di pooling 3x3 con uno stride di 2 su entrambe le dimensioni (righe e colonne).

    Puoi eseguire l'operazione di Average Pooling su una feature map ONNX utilizzando un framework di deep learning che supporta ONNX, come ad esempio PyTorch o TensorFlow. I dettagli specifici dell'implementazione possono variare a seconda del framework che stai utilizzando.
  */
}
fn op_lrn(){
    
}
impl <T> Into<ArrayD<T>> for TensorProto {
    fn into(self) -> ArrayD<T> {
        let shape = self.dims;
        match self.data_type {
            1 => {
                ArrayD::from_shape_vec(&shape.into_iter().collect_tuple().unwrap(), self.float_data).unwrap()
            }
            _ => {
                panic!("Type not defined")
            }
        }
    }
}