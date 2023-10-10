use std::sync::mpsc::{channel, Receiver, Sender};
use crate::onnx::{ModelProto, NodeProto, TensorProto};
use crate::utils::get_random_float_tensor;


pub struct OnnxRunningEnvironment<'a>{
    input_tensor: TensorProto,
    input_senders: Vec<Sender<TensorProto>>,
    model: ModelProto,
    node_io_vec: Vec<NodeIO<'a>>
}

impl OnnxRunningEnvironment<'_>{
    pub fn new(model: ModelProto) -> Self{
        let mut node_io_vec: Vec<NodeIO> = Vec::new();
        let graph = model.graph.unwrap();

        let mut input_senders: Vec<Sender<TensorProto>> = Vec::new();
        let input_node_name: &String = &graph.input.get(0).unwrap().name;
        for current_node in graph.node.iter() {
            let (sender, receiver) = channel();
            //per ogni valore NodeProto creo un elemento del vettore node_io_vec
            let new_node_io = NodeIO {
                senders: Vec::new(),//è il vettore dei sender che verrà generato in seguito dai nodi che hanno come input l'output del nodo in questione
                receiver,//receiver da cui leggere gli input
                node: current_node,
            };

            //si inserisce nei nodi che hanno come output gli input del nodo corrente il sender del nodo corrente
            for node_io in &mut node_io_vec {
                if node_io.node.output.iter().any(|x| current_node.input.contains(x)) {
                    node_io.senders.push(sender.clone());
                };
            }
            if current_node.input.contains(&input_node_name) {
                input_senders.push(sender.clone());
            }

            node_io_vec.push(new_node_io);
        }
        Self{
            input_tensor: get_random_float_tensor(vec![1,3,224,224]),
            input_senders,
            model,
            node_io_vec,
        }
    }
    pub fn run(&self) {
        //Invio il tensore di input della rete sui sender di input
        self.input_senders.iter().for_each(|first_sender| {
            println!("Start running using the following tensor as input: {:?}", self.input_tensor);
            first_sender.send(self.input_tensor.clone()).expect("Send of the input tensor failed!");
        });
        //println!("{:?}", input)


    }
}

#[derive(Debug)]
struct NodeIO<'a> {
    senders: Vec<Sender<TensorProto>>,
    receiver: Receiver<TensorProto>,
    node: &'a NodeProto,
}