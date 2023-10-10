mod onnx_running_environment;
mod utils;

mod onnx {
    include!("onnx.rs");
}

use onnx::{ModelProto};
use crate::onnx_running_environment::OnnxRunningEnvironment;

fn main() {
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    let parsed_proto: ModelProto =
        prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    let new_env = OnnxRunningEnvironment::new(parsed_proto);
    new_env.run();
}
