#[allow(unused_variables)]
#[allow(unused_imports)]
mod onnx {
    include!("onnx.rs");
}
use core::fmt;
use std::any::type_name;
use std::error::Error;
use std::fs::File;
use std::io::{ErrorKind, self, Write, Read};
use std::ops::Index;
use std::process::exit;
use ndarray::{arr2, Array, Array2, Array4, ArrayD, ArrayView,Dimension, IxDyn, s,Axis, Zip};
use onnx::tensor_proto::DataLocation;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::tract_itertools::Itertools;
use onnx::ModelProto;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;
use rand::prelude::*;


fn main() {
    let mut path_model:&str="";
    let mut path_testset:&str="";
    loop {
        println!("Onnx runtime");
        println!("scegli una rete:");
        println!("1. mobilenet");
        println!("2. resnet");
        println!("3. squeezenet");
        println!("4. googlenet");
        println!("5. fine");

        print!("Seleziona un'opzione: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).expect("Errore durante la lettura dell'input");

        // Rimuovi spazi e caratteri di nuova linea dall'input
        let choice = choice.trim();
        match choice {
            "1" => {path_model=&mobilenet_load();
                    loop {      
                        io::stdout().flush().unwrap();
                        println!("vuoi usare il test set di default ? (s/n)");    
                        let mut choice2 = String::new();
                        io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                        // Rimuovi spazi e caratteri di nuova linea dall'input
                        let choice2 = choice2.trim();
                        match choice2 {
                                "s" => {path_testset=&mobilenet_load_testset();break;},
                                "n" => {
                                    println!("implementare come inserire un test set diverso");
                                    break;
                                }
                            _ => println!("Scelta non valida. Riprova."),
                        }
                    
                    };break;},
            "2" => {path_model=&resnet_load();
                loop {        
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");    
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                            "s" => {path_testset=&resnet_load_testset();break;},
                            "n" => {
                                println!("implementare come inserire un test set diverso");
                                break;
                            }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                
                };break;},
            "3" => {path_model=&squeezenet_load();
                loop {        
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");    
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                            "s" => {path_testset=&squeezenet_load_testset();break;},
                            "n" => {
                                println!("implementare come inserire un test set diverso");
                                break;
                            }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                
                };break;},
            "4" => {path_model=&googlenet_load();
                loop {        
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");    
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                            "s" => {path_testset=&googlenet_load_testset();break;},
                            "n" => {
                                println!("implementare come inserire un test set diverso");
                                break;
                            }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                
                };break;},
            "5" => {
                println!("Uscita dal programma");
                break;
            }
            _ => println!("Scelta non valida. Riprova."),
        }
    }
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    //let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    if path_model.is_empty() || path_testset.is_empty() {
        exit(1)
    }
    let data = std::fs::read(path_model).expect("Failed to read ProtoBuf file");
    let parsed_proto: ModelProto = prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    // Use the parsed ProtoBuf data as needed
    //let i = parsed_proto.graph.unwrap().initializer.clone();
    //let j = i.clone();
    //let e =i.get(0).unwrap();
    //let es: ArrayD<f32>=into(e.clone()).unwrap();
    
    //println!("{:?}", type_of(es.clone()));
    //let matrix = Array4::from_shape_vec((64, 16,1,1), i.float_data).unwrap();
    //println!("{:?}", op_add(vec![i, j], vec![]));

}
fn read_input(input: &str) {
    // Path to your .pb file da concatenare
    let file_path = input;

    // Open the file
    let mut file = File::open(file_path).expect("Unable to open file");

    // Read the file contents into a Vec<u8>
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Unable to read file");

    // Deserialize the .pb file
    //let mut message = your_proto::YourMessage::new();  // Replace with your generated protobuf message type
    //message.merge_from_bytes(&buffer).expect("Unable to parse .pb file");

    // Access the data in the protobuf message
    //println!("Field value: {:?}", message.get_your_field());
}
fn mobilenet_load_testset()->&'static str{
    let path_testset="src/mobilenet/data_mobilenet/input_0.pb";
    return path_testset
}
fn mobilenet_load()->&'static str {
    let path_model="src/mobilenet/model.onnx";
    return path_model
}
fn googlenet_load_testset()->&'static str{
    let path_testset="src/googlenet/data_googlenet/input_0.pb";
    return path_testset
}
fn googlenet_load()->&'static str {
    let path_model="src/googlenet/model.onnx";
    return path_model
}
fn resnet_load_testset()->&'static str{
    let path_testset="src/resnet/data_resnet/input_0.pb";
    return path_testset
}
fn resnet_load()->&'static str {
    let path_model="src/resnet/model.onnx";
    return path_model
}
fn squeezenet_load_testset()->&'static str{
    let path_testset="src/squeezenet/data_squeezenet/input_0.pb";
    return path_testset
}
fn squeezenet_load()->&'static str {
    let path_model="src/squeezenet/model.onnx";
    return path_model
}

struct Operation{
    op_type: OperationType,
    input: Vec<TensorProto>,
    op_attributes: Vec<AttributeProto>
}

fn from<T>(array: ArrayD<T>, name : String) -> Result<TensorProto, OnnxError>
where
    T: Into<f32> + Into<f64> + Into<i32> + Into<i64>,
{
    let mut tensor = TensorProto {
        dims: array.shape().iter().map(|&x| x as i64).collect(),
        data_type: DataType::Undefined.into(),
        segment: None,
        name: name,
        doc_string: "".to_string(),
        data_location: DataLocation::Default.into(),
        float_data: Vec::new(),
        int32_data: Vec::new(),
        string_data: Vec::new(),
        int64_data: Vec::new(),
        raw_data: Vec::new(),
        external_data: Vec::new(),
        double_data: Vec::new(),
        uint64_data: Vec::new(),
    };
    match type_name::<T>() {
        "f32" => {
            tensor.data_type = DataType::Float.into();
            tensor.float_data = array.into_raw_vec().into_iter().map(|x| x.into()).collect();
            Ok(tensor)
        }
        _ => Err(OnnxError::new("Unsupported data type")),
    }
}

#[derive(Debug)]
struct OnnxError {
    message: String,
}

impl OnnxError {
    fn new(message: &str) -> OnnxError {
        OnnxError {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for OnnxError {
    fn description(&self) -> &str {
        &self.message
    }
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
fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}
/* 
fn perform_operation(op: Operation) -> Result<TensorProto, Error>{
    match op.op_type {
        OperationType::ADD => {
            //op_add(op.input.ge, )
        }
        OperationType::RELU => {
            //op_relu(op.input)
        }
        OperationType::EXP => {
            //op_exp
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

  
fn op_add(tensor1:TensorProto,tensor2 :TensorProto) -> Result<TensorProto, Error>{
    let array1=into(tensor1).unwrap();
    let array2=into(tensor2).unwrap();
    return Ok(from(array1+array2));
    
}
 
fn op_relu(input: TensorProto)->Result<TensorProto, Error>{
    let a: ArrayD<f32> = into(input).unwrap();
    for i in 0..a.len() {
        if a[i].clone() < 0.0 {
            a[i] = 0.0;
        }
    }
    return Ok(TensorProto::from(a));
}

fn op_exp(){
    todo!()
}
fn op_concat<T>(tensor1: TensorProto, tensor2: TensorProto) -> Result<ArrayD<T>, &'static str>
where
    T: From<f32> + Clone,
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

    if tensor1.dims != tensor2.dims {
        return Err("Le dimensioni dei tensori non sono compatibili per la concatenazione");
    }
    
    let shape: Vec<usize> = tensor1.dims.iter().map(|dim| *dim as usize).collect();
    let arr1= into(tensor1).unwrap();
    let arr2= into(tensor2).unwrap();   

    let concatenated_array = ndarray::stack(Axis(1), &[arr1.view(), arr2.view()]).unwrap();
    Ok(TensorProto::from(concatenated_array))

        

}


fn op_flatten(tensor: TensorProto) -> Result<TensorProto, OnnxError>

{
    // OK
    let array : ArrayD<f32> =into(tensor.clone()).unwrap();
    let len = array.len();

    // Usa into_shape per convertire l'array in uno monodimensionale
    let arr=array.into_shape(IxDyn(&[len])).unwrap();
    return Ok(from(arr,tensor.name).unwrap())

}
*/
/* 
fn op_reshape<T>(tensor: TensorProto, new_shape: Vec<usize>) -> Result<ArrayD<T>, &'static str>
where
    T: From<f32> + Clone,
{
    let total_elements: usize = tensor.dims.iter().map(|dim| *dim as usize).product();
    if total_elements != new_shape.iter().cloned().product() {
        return Err("La nuova forma non è compatibile con il numero totale di elementi nell'array");
    }
    let array: ArrayD<T> = Array::from_shape_vec(IxDyn(&new_shape), tensor.float_data.into_iter().map(T::from).collect()).unwrap();
    Ok(array)
       

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
/* 
fn op_conv<A, D>(input: &Array<A, D>, kernel: &Array<A, D>) -> Array<A, D>
where
    A: Clone + std::ops::Mul<Output = A> + std::ops::Add<Output = A> + Default,
    D: Dimension,
{
    // bias opzionale se non è presente salto l'operazione di aggiunta del bias
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

fn lrn<T>(input: &Array<T, ndarray::IxDyn>, alpha: T, beta: T, bias: T, size: usize) -> Array<T, ndarray::IxDyn>
where
    T: std::ops::AddAssign + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Copy,
{
    let mut output = Array::zeros(input.shape());

    // Calcola la normalizzazione per ogni punto nel tensore
    for index in input.indexed_iter() {
        let i = index.0;
        let val = *index.1;

        let start_idx = i.saturating_sub(size / 2);
        let end_idx = (i + size / 2).min(input.len_of(Axis(1)) - 1);

        let square_sum: T = input
            .slice_axis(Axis(1), start_idx..=end_idx)
            .iter()
            .map(|&x| x * x)
            .sum();

        let normalization = (alpha / (size as f32) * square_sum + bias).powf(beta);

        output[i] = val / normalization;
    }

    output
}
*/
    /*
    
L'operazione LRN (Local Response Normalization) in ONNX (Open Neural Network Exchange) è progettata per normalizzare i valori di un tensore lungo i canali in un'area locale. Questa normalizzazione viene eseguita calcolando la somma dei quadrati dei valori nei vicini di un dato punto lungo i canali e quindi normalizzando il valore originale di quel punto in base alla somma dei quadrati.

L'operazione LRN è comunemente utilizzata nelle reti neurali convoluzionali (CNN) e può contribuire a migliorare la capacità di generalizzazione del modello.
     In questo esempio:

InputTensor è il tensore di input.
OutputTensor è il tensore di output dopo l'applicazione dell'operazione LRN.
alpha, beta, e bias sono parametri che regolano la normalizzazione.
size rappresenta la dimensione dell'area locale su cui viene calcolata la normalizzazione.
Si noti che i parametri come alpha, beta, e size possono variare a seconda dell'implementazione specifica o della versione di ONNX. Assicurati di consultare la documentazione di ONNX o specifiche per i dettagli esatti.
     */
*/
fn into<T>(tensor: TensorProto) -> Result<ArrayD<T>, std::io::Error>
where 
    T:From<f32> 
{
    let shape: Vec<usize> = tensor.dims.iter().map(|dim| *dim as usize).collect();

    match tensor.data_type {
        1 => {
            //let float_data: Result<Vec<T>, _> = tensor.float_data.into_iter().map(T::from).collect();
            let data: Vec<T> = tensor
            .float_data
            .iter()
            .map(|&value| value.into())
            .collect();
            Ok(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
        }
        _ => Err(std::io::Error::new(std::io::ErrorKind::Other, "Unsupported data type")),
    }
}

