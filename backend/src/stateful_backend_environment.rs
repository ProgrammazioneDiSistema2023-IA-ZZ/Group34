#[allow(unused_imports)]
use crate::{
    onnx::{attribute_proto, AttributeProto, ModelProto, TensorProto},
    onnx_running_environment::OnnxModelEditor,
    utils::{read_tensor, read_model, get_path_from_ordinal, write_message, OnnxError},
};
use crate::{
    onnx_running_environment::OnnxRunningEnvironment,
    operations::utils::ndarray_to_tensor_proto,
    utils::{convert_img, results_to_string},
};
#[allow(unused_imports)]
use protobuf::Error;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    time::Instant, ptr::null,
};
#[allow(unused_imports)]
use tract_onnx::tract_core::model::Node;

struct StatefulPaths {
    folder: &'static str,
    model: &'static str,
    state: &'static str,
}

///all paths of the folder where the state is saved
const STATEFUL_PATHS: StatefulPaths = StatefulPaths {
    folder: "src/backend_state",
    model: "src/backend_state/model.onnx",
    state: "src/backend_state/state.json",
};

//stato salvato su file
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ServerState {
    pub model_name: String,
    pub default_input_path: String,
    pub default_output_pat: String,
}

impl ServerState {
    ///load from file or create new
    fn new() -> Self {
        let s = Self::new_empty();
        s.load_if_exists().unwrap()
    }

    fn new_empty() -> Self {
        ::std::default::Default::default()
    }

    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json_data = to_string(self)?;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(STATEFUL_PATHS.state)?;
        file.write_all(json_data.as_bytes())?;
        file.flush()?;
        Ok(())
    }

    //carica lo stato da file altrimenti non varia i campi
    fn load_if_exists(self) -> Result<Self, Box<dyn std::error::Error>> {
        if !Path::new(STATEFUL_PATHS.state).exists() {
            return Ok(self);
        }
        let mut file = File::open(STATEFUL_PATHS.state)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let data: ServerState = from_str(&contents)?;
        return Ok(data);
    }
}

#[derive(Serialize, Deserialize)]
pub struct NodeDto {
    node_name: String,
    input: Vec<String>,
    output: Vec<String>,
    operation_type: String,
    domain: String,
    attributes: Vec<(String, i32, String)>, //attribute_name, attribute_type, attribute_value
    doc_string: String,
}

#[allow(dead_code)]
pub fn start() -> std::io::Result<()> {
    let _ = fs::remove_dir_all(STATEFUL_PATHS.folder);
    fs::create_dir_all(STATEFUL_PATHS.folder)?;
    Ok(())
}

#[allow(dead_code)]
pub fn select_model(ordinal: usize) -> ModelProto {
    let mut state = ServerState::new();

    let path = get_path_from_ordinal(ordinal).unwrap();
    let model_proto = read_model(&path.model);
    //salvataggio del modello
    write_message(&model_proto, STATEFUL_PATHS.model).unwrap();
    state.model_name = path.model_name;
    state.default_input_path = path.test;
    state.default_output_pat = path.output;
    state.save().unwrap();

    model_proto
}

#[allow(dead_code)]
pub fn get_model() -> ModelProto {
    read_model(STATEFUL_PATHS.model)
}

#[allow(dead_code)]
pub fn get_state() -> ServerState {
    ServerState::new()
}

#[allow(dead_code)]
pub fn get_node(node_name: String) -> NodeDto {
    let model = get_model();
    let node_proto = model
        .graph
        .as_ref()
        .unwrap()
        .node
        .iter()
        .find(|n| n.name == node_name)
        .unwrap()
        .clone();
    NodeDto {
        node_name,
        input: node_proto.input,
        output: node_proto.output,
        operation_type: node_proto.op_type,
        domain: node_proto.domain,
        attributes: convert_to_vec(node_proto.attribute),
        doc_string: node_proto.doc_string,
    }
}

#[allow(dead_code)]
pub fn create_node(node_dto: NodeDto) -> ModelProto {
    let mut model = get_model();

    let attributes_proto = convert_to_attribute_proto(node_dto.attributes.clone());

    OnnxModelEditor::insert_node(
        node_dto.node_name.clone(),
        node_dto.input.clone(),
        node_dto.output.clone(),
        node_dto.operation_type.clone(),
        node_dto.domain.clone(),
        attributes_proto.clone(),
        node_dto.doc_string.clone(),
        &mut model,
    );
    write_message(&model, STATEFUL_PATHS.model).unwrap();
    model
}

#[allow(dead_code)]
pub fn modify_node(node_dto: NodeDto) -> ModelProto {
    let model = get_model();

    let attributes_proto = convert_to_attribute_proto(node_dto.attributes);

    let model = OnnxModelEditor::modify_node(
        node_dto.node_name,
        node_dto.input,
        node_dto.output,
        node_dto.operation_type,
        node_dto.domain,
        attributes_proto,
        node_dto.doc_string,
        model,
    );
    write_message(&model, STATEFUL_PATHS.model).unwrap();
    model
}

#[allow(dead_code)]
pub fn remove_node(node_name: String) -> ModelProto {
    let model = get_model();

    let model = OnnxModelEditor::remove_node(node_name, model);
    write_message(&model, STATEFUL_PATHS.model).unwrap();
    model
}

#[derive(Serialize)]
struct PredictionResult {
    predicted: String,
    expected: String,
    time: f64,
}

#[allow(dead_code, unused_mut)]
pub fn run(flag: bool, custom: bool, path: String) -> String {
    let mut state = ServerState::new();

    let model = get_model();
    let mut input_tensor: TensorProto = read_tensor(&state.default_input_path);
    if custom {
        let arr_d_img = convert_img(path.to_string());
        input_tensor = ndarray_to_tensor_proto::<f32>(arr_d_img, "data").unwrap();
    }
    let mut output_tensor: TensorProto = read_tensor(&state.default_output_pat);

    let start = Instant::now();
    let predicted_output = OnnxRunningEnvironment::new(model, input_tensor).run(flag);
    let duration = start.elapsed();

    // Create a PredictionResult struct
    let result = PredictionResult {
        predicted: results_to_string(predicted_output),
        expected: if custom {"".to_string()} else {results_to_string(output_tensor)},
        time: duration.as_secs_f64(),
    };

    // Serialize the struct to JSON
    let json_result = serde_json::to_string(&result).expect("Failed to serialize to JSON");

    return json_result;
}

fn convert_to_attribute_proto(attributes: Vec<(String, i32, String)>) -> Vec<AttributeProto> {
    let mut attributes_proto = Vec::new();
    for attr in attributes {
        let (attribute_name, attribute_type, attribute_value) = attr;

        let mut attrbute_proto = AttributeProto {
            ..Default::default()
        };
        attrbute_proto.name = attribute_name;
        attrbute_proto.r#type = attribute_type;
        match attribute_type {
            x if x == attribute_proto::AttributeType::Float as i32 => {
                attrbute_proto.f = attribute_value.parse().unwrap()
            }
            x if x == attribute_proto::AttributeType::Int as i32 => {
                attrbute_proto.i = attribute_value.parse().unwrap()
            }
            x if x == attribute_proto::AttributeType::String as i32 => {
                attrbute_proto.s = attribute_value.into_bytes()
            }
            x if x == attribute_proto::AttributeType::Floats as i32 => {
                attrbute_proto.floats = attribute_value
                    .split(";")
                    .map(|x| x.parse().unwrap())
                    .collect()
            }
            x if x == attribute_proto::AttributeType::Ints as i32 => {
                attrbute_proto.ints = attribute_value
                    .split(";")
                    .map(|x| x.parse().unwrap())
                    .collect()
            }
            x if x == attribute_proto::AttributeType::Strings as i32 => {
                attrbute_proto.strings = attribute_value
                    .split(";")
                    .map(|x| x.to_string().into_bytes())
                    .collect()
            }
            _ => {}
        }

        attributes_proto.push(attrbute_proto)
    }
    attributes_proto
}

fn convert_to_vec(attributes_proto: Vec<AttributeProto>) -> Vec<(String, i32, String)> {
    let mut attributes = Vec::new();
    for attr in attributes_proto {
        let mut t = (attr.name, attr.r#type, "".to_string());
        match attr.r#type {
            x if x == attribute_proto::AttributeType::Float as i32 => {
                t.2 = attr.f.to_string();
            }
            x if x == attribute_proto::AttributeType::Int as i32 => {
                t.2 = attr.i.to_string();
            }
            x if x == attribute_proto::AttributeType::String as i32 => {
                t.2 = String::from_utf8(attr.s).unwrap();
            }
            x if x == attribute_proto::AttributeType::Floats as i32 => {
                t.2 = attr
                    .floats
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(";");
            }
            x if x == attribute_proto::AttributeType::Ints as i32 => {
                t.2 = attr
                    .ints
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(";");
            }
            x if x == attribute_proto::AttributeType::Strings as i32 => {
                t.2 = attr
                    .strings
                    .iter()
                    .map(|x| String::from_utf8(x.clone()).unwrap())
                    .collect::<Vec<String>>()
                    .join(";");
            }
            _ => {}
        };

        attributes.push(t)
    }
    attributes
}
