use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use crate::onnx::{GraphProto, ModelProto, NodeProto, TensorProto};
use crate::utils::get_random_float_tensor;


pub struct OnnxRunningEnvironment {
    input_tensor: TensorProto,
    input_senders: Vec<Sender<TensorProto>>,
    output_receiver: Receiver<TensorProto>,
    model: ModelProto,
    node_io_vec: Vec<NodeIO>,
}

impl OnnxRunningEnvironment {
    pub fn new(model: ModelProto) -> Self {
        let mut node_io_vec: Vec<NodeIO> = Vec::new();
        let graph = model.clone().graph.unwrap();

        let mut input_senders: Vec<Sender<TensorProto>> = Vec::new();
        let mut output_receiver: Option<Receiver<TensorProto>> = None;
        let input_node_name: &String = &graph.input.get(0).unwrap().name;
        let output_node_name: &String = &graph.output.get(0).unwrap().name;
        let mut optional_receiver: Option<Receiver<TensorProto>> = None;
        for current_node in graph.node.into_iter() {
            let (sender, receiver) = channel();

            if current_node.input.contains(&input_node_name) {
                input_senders.push(sender.clone());
            }

            if current_node.output.contains(&output_node_name) {
                output_receiver = Some(receiver);
                optional_receiver = None;
            } else {
                optional_receiver = Some(receiver)
            }

            //per ogni valore NodeProto creo un elemento del vettore node_io_vec
            let new_node_io = NodeIO {
                senders: Vec::new(),//è il vettore dei sender che verrà generato in seguito dai nodi che hanno come input l'output del nodo in questione
                optional_receiver,//receiver da cui leggere gli input
                node: current_node.clone(),
                initializers: get_initializers(model.clone().graph.unwrap(), current_node.clone()),
            };

            //si inserisce nei nodi che hanno come output gli input del nodo corrente il sender del nodo corrente
            for node_io in &mut node_io_vec {
                if node_io.node.output.iter().any(|x| current_node.input.contains(x)) {
                    node_io.senders.push(sender.clone());
                };
            }

            node_io_vec.push(new_node_io);
        }
        Self {
            input_tensor: get_random_float_tensor(vec![1, 3, 224, 224]),
            input_senders,
            output_receiver: output_receiver.unwrap(),
            model,
            node_io_vec,
        }
    }
    pub fn run(&self) {
        //Invio il tensore di input della rete sui sender di input
        self.input_senders.iter().for_each(|first_sender| {
            println!("Start running using a random tensor of dims: {:?}", self.input_tensor.dims);
            first_sender.send(self.input_tensor.clone()).expect("Send of the input tensor failed!");
        });

        thread::scope(|s| {
            for current_node in self.node_io_vec.iter() {
                s.spawn(|| {
                    let NodeIO { senders, optional_receiver, node, initializers } = current_node;
                    if let Some(receiver) = optional_receiver {
                        let input_data = receiver.recv().unwrap();
                        //TODO Perform operations use input_data + initializers
                        let output_data = input_data;
                        for sender in senders.iter() {
                            sender.send(output_data.clone()).expect("TODO: panic message");
                        }
                    }
                });
            }
        });
        let result = self.output_receiver.recv();
        println!("The final result is a tensor of dims: {:?}", result.unwrap().dims)
    }
}

pub fn get_initializers(graph: GraphProto, node: NodeProto) -> Vec<TensorProto> {
    let mut return_inits: Vec<TensorProto> = vec![];
    if node.input.len() > 1 {
        let inits_from_graph = graph.initializer;
        let requested_inits_names: Vec<String> = node.clone().input.drain(1..).collect();
        println!("{:?}", requested_inits_names);
        requested_inits_names.iter().for_each(|requested_init_name| {
            for init in inits_from_graph.iter() {
                if init.name == requested_init_name.to_owned() {
                    return_inits.push(init.clone());
                    continue;
                };
            }
        });
    }
    return_inits
}

#[derive(Debug)]
struct NodeIO {
    senders: Vec<Sender<TensorProto>>,
    optional_receiver: Option<Receiver<TensorProto>>,
    node: NodeProto,
    initializers: Vec<TensorProto>,
}

unsafe impl Send for NodeIO {}

unsafe impl Sync for NodeIO {}