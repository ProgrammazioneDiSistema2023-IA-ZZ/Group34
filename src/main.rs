mod onnx {
    include!("onnx.rs");
}

use onnx::ModelProto;

fn main() {
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    let parsed_proto: ModelProto = prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    // Use the parsed ProtoBuf data as needed
    println!("{:?}", parsed_proto);
}