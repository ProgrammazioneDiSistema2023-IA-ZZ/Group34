use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use crate::onnx::{ModelProto, NodeProto, TensorProto};
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
                initializers: Vec::new()
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

        /*
        thread::scope(|s| {
            for current_node in self.node_io_vec.iter() {
                s.spawn(|| {
                    let NodeIO { senders, optional_receiver, .. } = current_node;
                    if let Some(receiver) = optional_receiver {
                        let input_data = receiver.recv().unwrap();
                        //TODO Perform operations
                        let output_data = input_data;
                        for sender in senders.iter() {
                            sender.send(output_data.clone()).expect("TODO: panic message");
                        }
                    }
                });
            }
        });
        */
        let result = self.output_receiver.recv();
        println!("The final result is a tensor of dims: {:?}", result.unwrap().dims)
    }
}

#[derive(Debug)]
struct NodeIO {
    senders: Vec<Sender<TensorProto>>,
    optional_receiver: Option<Receiver<TensorProto>>,
    node: NodeProto,
    initializers: Vec<TensorProto>,
}
unsafe impl Send for NodeIO {}

//fn find_and_do_operation(node_for_op:NodeProto,inputs: &Vec<&TensorProto>,initializers: Option<&Vec<&TensorProto>>,node: &NodeProto,){
/*
fn find_and_do_operation(node_for_op:NodeProto,nodeio:NodeIO){
    //op type è una string che indica  l'operazione da fare serve copiare e incollare il match che avevo fatto in protoc
    // gioele copialo e incollalo 
    let str_op=node_for_op.op_type.clone().as_str();
    println!("{}",node_for_op.op_type.clone());
    // match --> redirect alle operazioni in operations
    match str_op {
        "ADD" => add(inputs, nodeio.initializers,nodeio.node ),
        "RELU" => relu(inputs, nodeio.initializers, nodeio.node ),
        "EXP" => exp(inputs, nodeio.initializers, nodeio.node ),
        "CONCAT" => concat(inputs, nodeio.initializers, nodeio.node ),
        "FLATTEN" => flatten(inputs, nodeio.initializers, nodeio.node ),
        "RESHAPE" => reshape(inputs, nodeio.initializers, nodeio.node ),
        "CONV" => conv(inputs, nodeio.initializers, nodeio.node ),
        "MAXPOOL" => maxpool(inputs, nodeio.initializers, nodeio.node ),
        "BATCHNORM" => batchnorm(inputs, nodeio.initializers, nodeio.node ),
        "DROPOUT" => dropuot(inputs, nodeio.initializers, nodeio.node ),
        "SOFTMAX" => softmax(inputs, nodeio.initializers, nodeio.node ),
        "GEMM" => gemm(inputs, nodeio.initializers, nodeio.node ),
        "MATMUL" => matmul(inputs, nodeio.initializers, nodeio.node ),
        "REDUCESUM" => reducesum(inputs, nodeio.initializers, nodeio.node ),
        "GLOBALAVGPOOL" => globalavgsum(inputs, nodeio.initializers, nodeio.node ),
        "LRN" => lrn(inputs, nodeio.initializers, nodeio.node ),
        _ => println!("Operazione sconosciuta"),
    }

}*/


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