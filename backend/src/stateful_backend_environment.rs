use crate::{
    onnx::{attribute_proto, AttributeProto, ModelProto},
    onnx_running_environment::OnnxModelEditor,
    utils::{decode_message, get_path_from_ordinal, write_message},
    OnnxError,
};
use protobuf::Error;
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_string};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
};
use tract_onnx::model;

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
            .create_new(true)
            .write(true)
            .append(true)
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

#[allow(dead_code)]
pub fn start() -> std::io::Result<()> {
    fs::remove_dir_all(STATEFUL_PATHS.folder)?;
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

pub fn create_node(
    node_name: String,
    input: Vec<String>,
    output: Vec<String>,
    operation_type: String,
    domain: String,
    attributes: Vec<(String, attribute_proto::AttributeType, String)>,
    doc_string: String,
    name_node_before: String,
    name_node_after: String,
) -> ModelProto {
    let model = get_model();

    let mut attributes_proto = Vec::new();
    for attr in attributes {
        let (attribute_name, attribute_type, attribute_value) = attr;

        let mut attrbute_proto = AttributeProto {
            ..Default::default()
        };
        attrbute_proto.name = attribute_name;
        attrbute_proto.r#type = attribute_type as i32;
        match attribute_type {
            attribute_proto::AttributeType::Float => {
                attrbute_proto.f = attribute_value.parse().unwrap()
            }
            attribute_proto::AttributeType::Int => {
                attrbute_proto.i = attribute_value.parse().unwrap()
            }
            attribute_proto::AttributeType::String => {
                attrbute_proto.s = attribute_value.into_bytes()
            }
            _ => {}
        }

        attributes_proto.push(attrbute_proto)
    }

    OnnxModelEditor::insert_node(
        node_name,
        input,
        output,
        operation_type,
        domain,
        attributes_proto,
        doc_string,
        model
            .graph
            .as_ref()
            .unwrap()
            .node
            .iter()
            .find(|x| x.name == name_node_before)
            .map(|x| x.clone()),
        model
            .graph
            .as_ref()
            .unwrap()
            .node
            .iter()
            .find(|x| x.name == name_node_after)
            .map(|x| x.clone()),
        model
    )
}
