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
use ndarray::{arr2, Array, Array2, Array4, ArrayD, ArrayView, Dimension, IxDyn, s, Axis, Zip};
use onnx::tensor_proto::DataLocation;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::tract_itertools::Itertools;
use onnx::ModelProto;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;
use rand::prelude::*;


mod onnx_running_environment;
mod utils;


fn main() {
    let mut path_model: &str = "";
    let mut path_testset: &str = "";
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
            "1" => {
                path_model = &mobilenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &mobilenet_load_testset();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                };
                break;
            }
            "2" => {
                path_model = &resnet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &resnet_load_testset();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                };
                break;
            }
            "3" => {
                path_model = &squeezenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &squeezenet_load_testset();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                };
                break;
            }
            "4" => {
                path_model = &googlenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin().read_line(&mut choice2).expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &googlenet_load_testset();
                            break;
                        }
                        "n" => {
                            println!("implementare come inserire un test set diverso");
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                };
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
    let parsed_proto: ModelProto = prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");
    println!("starting Network...");
    let new_env = OnnxRunningEnvironment::new(parsed_proto);
    new_env.run();
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

fn googlenet_load_testset() -> &'static str {
    let path_testset = "src/googlenet/data_googlenet/input_0.pb";
    return path_testset;
}

fn googlenet_load() -> &'static str {
    let path_model = "src/googlenet/model.onnx";
    return path_model;
}

fn resnet_load_testset() -> &'static str {
    let path_testset = "src/resnet/data_resnet/input_0.pb";
    return path_testset;
}

fn resnet_load() -> &'static str {
    let path_model = "src/resnet/model.onnx";
    return path_model;
}

fn squeezenet_load_testset() -> &'static str {
    let path_testset = "src/squeezenet/data_squeezenet/input_0.pb";
    return path_testset;
}

fn squeezenet_load() -> &'static str {
    let path_model = "src/squeezenet/model.onnx";
    return path_model;
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
        T: From<f32>
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