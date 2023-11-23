#[allow(unused_imports)]
mod onnx {
    include!("onnx.rs");
}

use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;
use crate::onnx_running_environment::OnnxRunningEnvironment;
use crate::utils::get_random_float_tensor;
use core::fmt;
use ndarray::{arr2, s, Array, Array2, Array4, ArrayD, ArrayView, Axis, Dimension, IxDyn, Zip};
use onnx::tensor_proto::DataLocation;
use onnx::ModelProto;
use operations::utils::tensor_proto_to_ndarray;
use rand::prelude::*;
use std::any::type_name;
use std::error::Error;
use std::fs::File;
use std::io::{self, ErrorKind, Read, Write};
use std::ops::Index;
use std::process::exit;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::tract_itertools::Itertools;

mod onnx_running_environment;
mod operations;
mod utils;

fn main() {
    print_results(get_random_float_tensor(vec![1, 1000, 1, 1]));

    let mut path_model: &str = "";
    let mut path_testset: &str = "";
    let mut path_output: &str = "";
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
        io::stdin()
            .read_line(&mut choice)
            .expect("Errore durante la lettura dell'input");

        // Rimuovi spazi e caratteri di nuova linea dall'input
        let choice = choice.trim();
        match choice {
            "1" => {
                path_model = &mobilenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &mobilenet_load_testset();
                            path_output = &mobilenet_load_output();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "2" => {
                path_model = &resnet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &resnet_load_testset();
                            path_output = &resnet_load_output();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "3" => {
                path_model = &squeezenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &squeezenet_load_testset();
                            path_output = &squeezenet_load_output();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "4" => {
                path_model = &googlenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &googlenet_load_testset();
                            path_output = &googlenet_load_output();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
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
    let model_proto: ModelProto =
        prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    println!("Reading the inputs ...");
    let data = std::fs::read(path_testset).expect("Failed to read ProtoBuf file");
    let input_tensor: TensorProto =
        prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    println!("starting Network...");
    let new_env = OnnxRunningEnvironment::new(model_proto, input_tensor);

    let pred_out = new_env.run(); //predicted output

    let data = std::fs::read(path_output).expect("Failed to read ProtoBuf file");
    let output_tensor: TensorProto =
        prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    println!("Predicted classes:");
    print_results(pred_out);
    println!("Ground truth classes:");
    print_results(output_tensor);
}

fn print_results(tensor: TensorProto) {
    let data = tensor_proto_to_ndarray::<f32>(&tensor).unwrap();

    for element in data
        .iter()
        .enumerate()
        .sorted_by(|a, b| b.1.total_cmp(a.1))
        .take(3)
    {
        print!("|Class n:{} Value:{}| ", element.0, element.1);
    }
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

    // If not preprocessing, proceed with model loading and inference
}

fn mobilenet_load_testset() -> &'static str {
    let path_testset = "src/mobilenet/data_mobilenet/input_0.pb";
    return path_testset;
}

fn mobilenet_load() -> &'static str {
    let path_model = "src/mobilenet/model.onnx";
    return path_model;
}

fn mobilenet_load_output() -> &'static str {
    let path_output = "src/mobilenet/data_mobilenet/output_0.pb";
    return path_output;
}

fn googlenet_load_testset() -> &'static str {
    let path_testset = "src/googlenet/data_googlenet/input_0.pb";
    return path_testset;
}

fn googlenet_load() -> &'static str {
    let path_model = "src/googlenet/model.onnx";
    return path_model;
}
fn googlenet_load_output() -> &'static str {
    let path_output = "src/googlenet/data_googlenet/output_0.pb";
    return path_output;
}
fn resnet_load_testset() -> &'static str {
    let path_testset = "src/resnet/data_resnet/input_0.pb";
    return path_testset;
}

fn resnet_load() -> &'static str {
    let path_model = "src/resnet/model.onnx";
    return path_model;
}

fn resnet_load_output() -> &'static str {
    let path_output = "src/resnet/data_resnet/output_0.pb";
    return path_output;
}

fn squeezenet_load_testset() -> &'static str {
    let path_testset = "src/squeezenet/data_squeezenet/input_0.pb";
    return path_testset;
}

fn squeezenet_load() -> &'static str {
    let path_model = "src/squeezenet/model.onnx";
    return path_model;
}

fn squeezenet_load_output() -> &'static str {
    let path_output = "src/squeezenet/data_squeezenet/output_0.pb";
    return path_output;
}
struct Operation {
    op_type: OperationType,
    input: Vec<TensorProto>,
    op_attributes: Vec<AttributeProto>,
}

fn from<T>(array: ArrayD<T>, name: String) -> Result<TensorProto, OnnxError>
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
pub enum OnnxError {
    /// Indicates that a required attribute was not found.
    ///
    /// The contained `String` provides the name or identifier of the missing attribute.
    AttributeNotFound(String),

    /// Represents generic internal errors that might occur during processing.
    ///
    /// The contained `String` provides a description or message detailing the nature of the internal error.
    InternalError(String),

    /// Indicates an error that occurred during data type conversion.
    ///
    /// The contained `String` provides additional information about the conversion that failed.
    ConversionError(String),

    /// Represents an error where an operation or functionality is not supported.
    ///
    /// The contained `String` provides details about the unsupported operation.
    UnsupportedOperation(String),

    /// Indicates a mismatch between expected and actual tensor shapes.
    ///
    /// The contained `String` provides details about the shape mismatch, such as the expected vs. actual dimensions.
    ShapeMismatch(String),

    /// Represents an error where an expected input tensor or data is missing.
    ///
    /// The contained `String` provides details about the missing input.
    MissingInput(String),

    /// Indicates an error due to invalid data or values.
    ///
    /// The contained `String` provides details about the nature of the invalid data.
    InvalidValue(String),

    /// Indicates an error related to tensor shape computations.
    ///
    /// The contained `String` provides details about the shape computation error.
    ShapeError(String),
}

impl OnnxError {
    fn new(message: &str) -> OnnxError {
        OnnxError::InternalError(message.to_string())
    }
}

// impl fmt::Display for OnnxError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.message)
//     }
// }

// impl Error for OnnxError {
//     fn description(&self) -> &str {
//         &self.message
//     }
// }
impl TensorProto {
    pub fn new() -> TensorProto {
        ::std::default::Default::default()
    }
}

enum OperationType {
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
    LRN,
}

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

fn into<T>(tensor: TensorProto) -> Result<ArrayD<T>, std::io::Error>
where
    T: From<f32>,
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
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unsupported data type",
        )),
    }
}
