mod onnx {
    include!("onnx.rs");
}

use std::{
    collections::HashMap,
    sync::mpsc::{channel, Receiver, Sender},
};

use onnx::{ModelProto, NodeProto, TensorProto};

fn main() {
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    let parsed_proto: ModelProto =
        prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");
}

fn initialize(model: ModelProto) {
    let mut node_io_vec: Vec<NodeIO> = Vec::new();

    for node in &model.graph.unwrap().node {
        let (sender, receiver) = channel();
        //per ogni valore NodeProto creo un elemento del vettore node_io_vec
        let node_io = NodeIO {
            senders: Vec::new(),//è il vettore dei sender che verrà generato in seguito dai nodi che hanno come input l'output del nodo in questione
            receiver: receiver,//receiver da cui leggere gli input
            node: node,
        };

        //si inserisce nei nodi che hanno come output gli input del nodo corrente il sender del nodo corrente
        for node_io in & mut node_io_vec {
            if node_io.node.output.iter().any(|x| node.input.contains(x)) {
                node_io.senders.push(sender.clone());
            };
        }

        node_io_vec.push(node_io);
    }
}

struct NodeIO<'a> {
    senders: Vec<Sender<TensorProto>>,
    receiver: Receiver<TensorProto>,
    node: &'a NodeProto,
}
