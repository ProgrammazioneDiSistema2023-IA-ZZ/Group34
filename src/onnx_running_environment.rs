use crate::onnx::{self, GraphProto, ModelProto, NodeProto, TensorProto};
use crate::operations::add;
use crate::operations::batch_norm;
use crate::operations::concat;
use crate::operations::conv;
use crate::operations::dropout;
use crate::operations::exp;
use crate::operations::flatten;
use crate::operations::gemm;
use crate::operations::global_average_pool::{self, globalavgpool};
use crate::operations::lrn;
use crate::operations::matmul;
use crate::operations::maxpool;
use crate::operations::reducesum;
use crate::operations::relu;
use crate::operations::reshape;
use crate::operations::softmax;
use crate::utils::get_random_float_tensor;
use crate::OnnxError;
use image::flat::Error;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;
use tract_onnx::prelude::tract_itertools::Itertools;
pub struct OnnxRunningEnvironment {
    input_tensor: TensorProto,
    input_senders: Vec<Sender<TensorProto>>,
    output_receiver: Receiver<TensorProto>,
    model: ModelProto,
    node_io_vec: Vec<NodeIO>,
}

impl OnnxRunningEnvironment {
    pub fn new(model: ModelProto, input_tensor: TensorProto) -> Self {
        let mut node_io_vec: Vec<NodeIO> = Vec::new();
        let mut graph = model.clone().graph.unwrap();
        let mut input_tensor_clone=input_tensor.clone();
        let mut input_senders: Vec<Sender<TensorProto>> = Vec::new();
        let (output_sender, output_receiver) = channel();
        let input_node_name: &String = &graph.input.get(0).unwrap().name;
        let output_node_name: &String = &graph.output.get(0).unwrap().name;
        let mut optional_receiver: Option<Receiver<TensorProto>> = None;
        input_tensor_clone.name=input_node_name.clone();
        for current_node in graph.node.into_iter() {
            let (sender, receiver) = channel();
            let mut senders = Vec::new();

            if current_node.input.contains(&input_node_name) {
                input_senders.push(sender.clone());
            }

            if current_node.output.contains(&output_node_name) {
                senders.push(output_sender.clone());
            }
            optional_receiver = Some(receiver);
            let mut current_node_clone=current_node.clone();
            if(current_node.name==""){
                current_node_clone.name="name_node".to_string();
            }
            //per ogni valore NodeProto creo un elemento del vettore node_io_vec
            let new_node_io = NodeIO {
                senders, //è il vettore dei sender che verrà generato in seguito dai nodi che hanno come input l'output del nodo in questione
                optional_receiver,   //receiver da cui leggere gli input
                node: current_node_clone.clone(),
                initializers: get_initializers(model.clone().graph.unwrap(), current_node.clone()),
            };

            //si inserisce nei nodi che hanno come output gli input del nodo corrente il sender del nodo corrente
            for node_io in &mut node_io_vec {
                if node_io
                    .node
                    .output
                    .iter()
                    .any(|x| current_node.input.contains(x))
                {
                    node_io.senders.push(sender.clone());
                };
            }

            node_io_vec.push(new_node_io);
        }
        Self {
            input_tensor:input_tensor_clone,
            input_senders,
            output_receiver,
            model,
            node_io_vec,
        }
    }
    pub fn run(&self) -> TensorProto {
        let start = Instant::now();
        //Invio il tensore di input della rete sui sender di input
        self.input_senders.iter().for_each(|first_sender| {
            println!(
                "Start running using a tensor of dims: {:?} and name: {:?}",
                self.input_tensor.dims, self.input_tensor.name
            );
            first_sender
                .send(self.input_tensor.clone())
                .expect("Send of the input tensor failed!");
        });

        thread::scope(|s| {
            for current_node in self.node_io_vec.iter() {
                s.spawn(|| {
                    let NodeIO {
                        senders,
                        optional_receiver,
                        node,
                        initializers,
                    } = current_node;
                    if let Some(receiver) = optional_receiver {
                        let inputs = get_inputs(receiver, node, initializers);
                        let output_result = find_and_do_operation(node, initializers.clone(), inputs.clone());
                        match output_result {
                            Ok(mut output_data) => {
                                output_data.name = String::from(node.clone().output.first().unwrap());
                                for sender in senders.iter() {
                                    sender
                                        .send(output_data.clone())
                                        .expect("TODO: panic message");
                                }
                            }
                            Err(e) => {
                                println!("Error: {:?}", e)
                            }
                        }
                    }
                });
            }
        });
        let result = self.output_receiver.recv().unwrap();

        let duration = start.elapsed();
        println!("\nRun ended - elapsed: {:?}", duration);
        println!(
            "The final result is a tensor of dims: {:?} and name {:?}",
            result.dims,
            result.name
        );
        result
    }
}


pub fn get_inputs(
    receiver: &Receiver<TensorProto>,
    node: &NodeProto,
    initializers: &Vec<TensorProto>,
) -> Vec<TensorProto> {
    let mut input = receiver.recv().unwrap();
    let inputs_and_initializers_to_read_names = node.input.clone();
    let outputs_and_initializers_to_read_names = input.clone();
    let initializers_names: Vec<String> = initializers.iter().map(|i| i.name.clone()).collect();
    let mut inputs_names = vec![input.name.clone()];
    let mut inputs = vec![input.clone()];
    while !is_input_reading_finished(
        inputs_and_initializers_to_read_names.clone(),
        initializers_names.clone(),
        inputs_names.clone(),
    ) {
        input = receiver.recv().unwrap();
        inputs_names.push(input.name.clone());
        inputs.push(input.clone())
    }
    inputs
}

pub fn is_input_reading_finished(
    inputs_and_initializers_to_read_names: Vec<String>,
    initializers_names: Vec<String>,
    inputs_names: Vec<String>,
) -> bool {
    let mut elements_found = 0;
    inputs_names.iter().for_each(|input_name| {
        if let Some(n) = inputs_and_initializers_to_read_names
            .iter()
            .find(|i| i == &input_name)
        {
            elements_found += 1;
        }
    });
    initializers_names.iter().for_each(|initializer_names| {
        if let Some(n) = inputs_and_initializers_to_read_names
            .iter()
            .find(|i| i == &initializer_names)
        {
            elements_found += 1;
        }
    });
    inputs_and_initializers_to_read_names.len() == elements_found
}

pub fn get_initializers(graph: GraphProto, node: NodeProto) -> Vec<TensorProto> {
    let mut return_inits: Vec<TensorProto> = vec![];
    if node.input.len() > 1 {
        let inits_from_graph = graph.initializer;
        let requested_inits_names: Vec<String> = node.clone().input.drain(1..).collect();
        requested_inits_names
            .iter()
            .for_each(|requested_init_name| {
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

fn find_and_do_operation(
    node_for_op: &NodeProto,
    initializers: Vec<TensorProto>,
    inputs: Vec<TensorProto>,
) -> Result<TensorProto, OnnxError> {
    let str_op = node_for_op.op_type.to_uppercase();
    println!("Node: {} - Operation:{}", node_for_op.name, str_op);

    // //Gli input vengono inseriti dal nodo precedente nel receiver
    // let recv = nodeio.optional_receiver.ok_or(OnnxError::InternalError("[RUN op] Receiver is None. It should be Some in order to execute the operation".to_string()))?;
    // let inputs = nodeio.optional_receiver.unwrap().recv().map_err(|error| OnnxError::MissingInput("[RUN op] The Sender where dropped before sending inputs".to_string()))?;
    // // match --> redirect alle operazioni in operations
    match str_op.as_str() {
        "ADD" => add(inputs, initializers, node_for_op),
        "RELU" => relu(inputs, node_for_op),
        "EXP" => exp(inputs, node_for_op),
        "CONCAT" => concat(&inputs, node_for_op), // use initializers ?
        "FLATTEN" => flatten(inputs, node_for_op),
        "RESHAPE" => reshape(inputs, initializers, node_for_op), //    input: ArrayViewD<f32>,shape: &Vec<isize>,allow_zero: i64,
        "CONV" => conv(inputs, initializers, node_for_op),
        "MAXPOOL" => maxpool(inputs, node_for_op),
        "BATCHNORMALIZATION" => batch_norm(inputs, initializers, node_for_op),
        "DROPOUT" => dropout(inputs, initializers, node_for_op),
        "SOFTMAX" => softmax(inputs, node_for_op),
        "GEMM" => gemm(inputs, initializers, node_for_op),
        "MATMUL" => matmul(inputs, initializers, node_for_op),
        "REDUCESUM" => reducesum(inputs, node_for_op),
        //"AVERAGEPOOL" => globalavgpool(inputs, node_for_op), 
        // eliminare squeezenet --> evitiamo avgpooling
        "GLOBALAVERAGEPOOL" => globalavgpool(inputs, node_for_op),
        "LRN" => lrn(inputs, node_for_op),
        _ => Err(OnnxError::UnsupportedOperation("Operazione non supportata".to_string())),
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
