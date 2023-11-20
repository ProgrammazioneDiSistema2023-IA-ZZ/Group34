use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use image::flat::Error;
use tract_onnx::prelude::tract_itertools::Itertools;
use crate::OnnxError;
use crate::onnx::{self, GraphProto, ModelProto, NodeProto, TensorProto};
use crate::utils::get_random_float_tensor;
use crate::operations::relu;
use crate::operations::concat;
use crate::operations::exp;
use crate::operations::add;
use crate::operations::flatten;
use crate::operations::reshape;
use crate::operations::conv;
use crate::operations::maxpool;
use crate::operations::batch_norm;
use crate::operations::dropout;
use crate::operations::softmax;
use crate::operations::gemm;
use crate::operations::matmul;
use crate::operations::reducesum;
use crate::operations::lrn;
use crate::operations::global_average_pool;
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
        //println!("{?}",graph.unwrap());
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
            input_tensor: get_random_float_tensor(vec![1, 3, 224, 224]), //TODO sostituire con l'effettivo tensore in input
            input_senders,
            output_receiver: output_receiver.unwrap(),
            model,
            node_io_vec,
        }
    }
    pub fn run(&self) {
        //Invio il tensore di input della rete sui sender di input
        self.input_senders.iter().for_each(|first_sender| {
            println!("Start running using a random tensor of dims: {:?} and name: {:?}", self.input_tensor.dims, self.input_tensor.name);
            first_sender.send(self.input_tensor.clone()).expect("Send of the input tensor failed!");
        });

        thread::scope(|s| {
            for current_node in self.node_io_vec.iter() {
                s.spawn(|| {
                    let NodeIO { senders, optional_receiver, node, initializers } = current_node;
                    if let Some(receiver) = optional_receiver {
                        let inputs = get_inputs(receiver, node, initializers);

                        //TODO To perform operations use inputs + initializers, note that if inputs has only one number you need to extract it from to Vec before giving it to the operation function
                        let mut output_data = inputs.first().unwrap().clone(); //replace the right end side with the operation call

                        output_data.name = String::from(node.clone().output.first().unwrap());
                        for sender in senders.iter() {
                            sender.send(output_data.clone()).expect("TODO: panic message");
                        }
                    }
                });
            }
        });
        let result = self.output_receiver.recv();
        println!("The final result is a tensor of dims: {:?} and name {:?}", result.clone().unwrap().dims, result.unwrap().name)
    }
}

pub fn get_inputs(receiver: &Receiver<TensorProto>, node: &NodeProto, initializers: &Vec<TensorProto>) -> Vec<TensorProto>{
    let mut input = receiver.recv().unwrap();
    let inputs_and_initializers_to_read_names = node.input.clone();
    let initializers_names: Vec<String> = initializers.iter().map(|i|i.name.clone()).collect();
    let mut inputs_names = vec!(input.clone().name);
    let mut inputs = vec!(input.clone());
    while !is_input_reading_finished(inputs_and_initializers_to_read_names.clone(), initializers_names.clone(), inputs_names.clone()) {
        input = receiver.recv().unwrap();
        inputs_names.push(input.name.clone());
        inputs.push(input.clone())
    }
    inputs
}

pub fn is_input_reading_finished(inputs_and_initializers_to_read_names: Vec<String>, initializers_names: Vec<String>, inputs_names: Vec<String>) -> bool{
    let mut elements_found = 0;
    inputs_names.iter().for_each(|input_name|{
        if let Some(n) = inputs_and_initializers_to_read_names.iter().find(|i|i==&input_name){
            println!("N1 {:?}", n);
            elements_found+=1;
        }
    });
    initializers_names.iter().for_each(|initializer_names|{
        if let Some(n) = inputs_and_initializers_to_read_names.iter().find(|i|i==&initializer_names){
            println!("N2 {:?}", n);
            elements_found+=1;
        }
    });
    inputs_and_initializers_to_read_names.len() == elements_found
}

pub fn get_initializers(graph: GraphProto, node: NodeProto) -> Vec<TensorProto> {
    let mut return_inits: Vec<TensorProto> = vec![];
    if node.input.len() > 1 {
        let inits_from_graph = graph.initializer;
        let requested_inits_names: Vec<String> = node.clone().input.drain(1..).collect();
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

fn find_and_do_operation(node_for_op:NodeProto,nodeio:NodeIO) -> Result<(),OnnxError>{
    let str_op=node_for_op.op_type.clone().as_str();
    println!("{}",node_for_op.op_type.clone());
    
    //Gli input vengono inseriti dal nodo precedente nel receiver
    let recv = nodeio.optional_receiver.ok_or(OnnxError::InternalError("[RUN op] Receiver is None. It should be Some in order to execute the operation".to_string()))?;
    let inputs = nodeio.optional_receiver.unwrap().recv().map_err(|error| OnnxError::MissingInput("[RUN op] The Sender where dropped before sending inputs".to_string()))?;
    // match --> redirect alle operazioni in operations
    match str_op {
        "ADD" => add(inputs, nodeio.initializers.clone(),&nodeio.node.clone() ),
        "RELU" => relu(&inputs, &nodeio.node ),
        "EXP" => exp(&inputs, &nodeio.node ),
        "CONCAT" => concat(&inputs, &nodeio.node ),
        "FLATTEN" => flatten(&inputs, &nodeio.node ),
        "RESHAPE" => reshape(inputs, nodeio.initializers, &nodeio.node ),//    input: ArrayViewD<f32>,shape: &Vec<isize>,allow_zero: i64,
        "CONV" => conv(&inputs, nodeio.initializers, &nodeio.node ),
        "MAXPOOL" => maxpool(&inputs, &nodeio.node ),
        "BATCHNORM" => batch_norm(&inputs, nodeio.initializers, &nodeio.node ),
        "DROPOUT" => dropout(&inputs, nodeio.initializers, &nodeio.node ),
        "SOFTMAX" => softmax(&inputs, &nodeio.node ),
        "GEMM" => gemm(inputs, nodeio.initializers, &nodeio.node ),
        "MATMUL" => matmul(inputs, nodeio.initializers, &nodeio.node ),
        "REDUCESUM" => reducesum(&inputs, &nodeio.node ),
        "GLOBALAVGPOOL" => globalavgpool(inputs, nodeio.node ),
        "LRN" => lrn(&inputs, &nodeio.node ),
        _ => println!("Operazione sconosciuta"),
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