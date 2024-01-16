#[allow(unused_imports)]
use crate::{
    onnx::{attribute_proto, AttributeProto, ModelProto, TensorProto},
    onnx_running_environment::OnnxModelEditor,
    utils::{decode_message, get_path_from_ordinal, write_message, OnnxError},
};
use crate::{onnx_running_environment::OnnxRunningEnvironment, utils::results_to_string};
#[allow(unused_imports)]
use protobuf::Error;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
};
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
            .open(STATEFUL_PATHS.state)?;

        file.write_all(json_data.as_bytes())?;
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

#[derive(Serialize)]
pub struct NodeDto {
    node_name: String,
    input: Vec<String>,
    output: Vec<String>,
    operation_type: String,
    domain: String,
    attributes: Vec<(String, i32, String)>, //attribute_name, attribute_type, attribute_value
    doc_string: String,
    name_node_before: String,
    name_node_after: String,
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
    let model_proto: ModelProto = decode_message(&path.model);
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
    decode_message(STATEFUL_PATHS.model)
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
        name_node_before: "".to_string(),
        name_node_after: "".to_string(),
    }
}



#[allow(dead_code)]
pub fn create_node(node_dto: NodeDto) -> ModelProto {
    let model = get_model();

    let attributes_proto = convert_to_attribute_proto(node_dto.attributes.clone());

    OnnxModelEditor::insert_node(
        node_dto.node_name.clone(),
        node_dto.input.clone(),
        node_dto.output.clone(),
        node_dto.operation_type.clone(),
        node_dto.domain.clone(),
        attributes_proto.clone(),
        node_dto.doc_string.clone(),
        model
            .graph
            .as_ref()
            .unwrap()
            .node
            .iter()
            .find(|x| x.name == node_dto.name_node_before.clone())
            .map(|x| x.clone()),
        model
            .graph
            .as_ref()
            .unwrap()
            .node
            .iter()
            .find(|x| x.name == node_dto.name_node_after)
            .map(|x| x.clone()),
        model,
    )
}

#[allow(dead_code)]
pub fn modify_node(node_dto: NodeDto) -> ModelProto {
    let model = get_model();

    let attributes_proto = convert_to_attribute_proto(node_dto.attributes);

    OnnxModelEditor::modify_node(
        node_dto.node_name,
        node_dto.input,
        node_dto.output,
        node_dto.operation_type,
        node_dto.domain,
        attributes_proto,
        node_dto.doc_string,
        model,
    )
}

#[allow(dead_code)]
pub fn remove_node(node_name: String) -> ModelProto {
    let model = get_model();

    OnnxModelEditor::remove_node(node_name, model)
}

#[allow(dead_code)]
pub fn run() -> String {
    let mut state = ServerState::new();

    let model = get_model();
    let mut input_tensor: TensorProto = decode_message(&state.default_input_path);
    let mut output_tensor: TensorProto = decode_message(&state.default_output_pat);

    let predicted_output = OnnxRunningEnvironment::new(model, input_tensor).run(true);

    return format!(
        "Predicted classes: \n{}\n\n Expected output: \n{}",
        results_to_string(predicted_output),
        results_to_string(output_tensor)
    );
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
            x if x == attribute_proto::AttributeType::Float as i32  => {
                t.2 = attr.f.to_string();
            }
            x if x == attribute_proto::AttributeType::Int as i32  => {
                t.2 = attr.i.to_string();
            }
            x if x == attribute_proto::AttributeType::String as i32  => {
                t.2 = String::from_utf8(attr.s).unwrap();
            }
            _ => {}
        };

        attributes.push(t)
    }
    attributes
}
