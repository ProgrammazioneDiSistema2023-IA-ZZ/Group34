use crate::onnx::{
    AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    SparseTensorProto, StringStringEntryProto, TensorAnnotation, TensorProto, TrainingInfoProto,
    ValueInfoProto,
};
use crate::operations::add;
use crate::operations::batch_norm;
use crate::operations::concat;
use crate::operations::conv;
use crate::operations::dropout;
use crate::operations::exp;
use crate::operations::flatten;
use crate::operations::gemm;
use crate::operations::global_average_pool::globalavgpool;
use crate::operations::lrn;
use crate::operations::matmul;
use crate::operations::maxpool;
use crate::operations::reducesum;
use crate::operations::relu;
use crate::operations::reshape;
use crate::operations::softmax;

use crate::utils::OnnxError;

use std::collections::{HashMap, LinkedList};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;
#[allow(unused_imports)]
use tract_onnx::prelude::tract_itertools::Itertools;
pub struct OnnxRunningEnvironment {
    input_tensor: TensorProto,
    input_senders: Vec<Sender<TensorProto>>,
    output_receiver: Receiver<TensorProto>,
    model: ModelProto,
    node_io_vec: Vec<NodeIO>,
}
#[allow(unused)]
impl OnnxRunningEnvironment {
    pub fn new(model: ModelProto, input_tensor: TensorProto) -> Self {
        let mut node_io_vec: Vec<NodeIO> = Vec::new();
        let graph = model.clone().graph.unwrap();
        let mut input_tensor_clone = input_tensor.clone();
        let mut input_senders: Vec<Sender<TensorProto>> = Vec::new();
        let (output_sender, output_receiver) = channel();
        let input_node_name: &String = &graph.input.get(0).unwrap().name;
        let output_node_name: &String = &graph.output.get(0).unwrap().name;
        let mut optional_receiver: Option<Receiver<TensorProto>> = None;
        input_tensor_clone.name = input_node_name.clone();
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
            let mut current_node_clone = current_node.clone();
            //per ogni valore NodeProto creo un elemento del vettore node_io_vec
            let new_node_io = NodeIO {
                senders, //è il vettore dei sender che verrà generato in seguito dai nodi che hanno come input l'output del nodo in questione
                optional_receiver, //receiver da cui leggere gli input
                node: current_node_clone.clone(),
                initializers: Self::get_initializers(
                    model.clone().graph.unwrap(),
                    current_node.clone(),
                ),
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
            input_tensor: input_tensor_clone,
            input_senders,
            output_receiver,
            model,
            node_io_vec,
        }
    }
    pub fn run(&self, flag_par: bool) -> TensorProto {
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
                s.spawn(move || {
                    let NodeIO {
                        senders,
                        optional_receiver,
                        node,
                        initializers,
                    } = current_node;
                    if let Some(receiver) = optional_receiver {
                        let inputs = Self::get_inputs(receiver, node, initializers);
                        let output_result = find_and_do_operation(
                            node,
                            initializers.clone(),
                            inputs.clone(),
                            flag_par,
                        );
                        match output_result {
                            Ok(mut output_data) => {
                                output_data.name =
                                    String::from(node.clone().output.first().unwrap());
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
            result.dims, result.name
        );
        result
    }
    pub fn run_sequential(&self, flag_par: bool) -> TensorProto {
        // Capture the current time before running the model
        // model: &ModelProto, input_tensor: TensorProto
        let start = Instant::now();
        let model = self.model.clone();
        let input_tensor = self.input_tensor.clone();
        // Extract the graph from the model.
        let graph = model.graph.unwrap();
        println!(
            "Start running sequential using a tensor of dims: {:?} and name: {:?}",
            self.input_tensor.dims, self.input_tensor.name
        );
        // Initialize a map to hold the tensors for each node's input.
        let mut input_map: HashMap<String, TensorProto> = HashMap::new();
        input_map.insert(graph.input[0].name.clone(), input_tensor);

        // Map the initializers by their names for easy lookup.
        let initializers_map: HashMap<String, TensorProto> = graph
            .initializer
            .iter()
            .map(|tensor_proto| (tensor_proto.name.clone(), tensor_proto.clone()))
            .collect();

        // Iterate over each node in the graph.
        for node in &graph.node {
            // Gather the inputs for the current node.
            let node_inputs: Vec<_> = node
                .input
                .iter()
                .filter_map(|name| input_map.get(name).cloned())
                .collect();

            // Gather the initializers for the current node.
            let node_initializers: Vec<_> = node
                .input
                .iter()
                .filter_map(|name| initializers_map.get(name).cloned())
                .collect();

            let output_tensor =
                find_and_do_operation(node, node_initializers, node_inputs, flag_par)
                    .expect("Failed to run node");

            let output_name = output_tensor.name.to_string();
            // Store the output tensor so it can be used as input for subsequent nodes.
            input_map.insert(output_name, output_tensor);
        }

        let duration = start.elapsed();
        println!("duration : {:?})\n", duration);

        // Return the output tensor for the entire model.
        input_map
            .get(&graph.output[0].name)
            .expect("Output tensor not found")
            .clone()
    }

    fn get_inputs(
        receiver: &Receiver<TensorProto>,
        node: &NodeProto,
        initializers: &Vec<TensorProto>,
    ) -> Vec<TensorProto> {
        let mut input = receiver.recv().unwrap();
        let inputs_and_initializers_to_read_names = node.input.clone();
        let _outputs_and_initializers_to_read_names = input.clone();
        let initializers_names: Vec<String> = initializers.iter().map(|i| i.name.clone()).collect();
        let mut inputs_names = vec![input.name.clone()];
        let mut inputs = vec![input.clone()];
        while !Self::is_input_reading_finished(
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

    fn is_input_reading_finished(
        inputs_and_initializers_to_read_names: Vec<String>,
        initializers_names: Vec<String>,
        inputs_names: Vec<String>,
    ) -> bool {
        let mut elements_found = 0;
        inputs_names.iter().for_each(|input_name| {
            if let Some(_n) = inputs_and_initializers_to_read_names
                .iter()
                .find(|i| i == &input_name)
            {
                elements_found += 1;
            }
        });
        initializers_names.iter().for_each(|initializer_names| {
            if let Some(_n) = inputs_and_initializers_to_read_names
                .iter()
                .find(|i| i == &initializer_names)
            {
                elements_found += 1;
            }
        });
        inputs_and_initializers_to_read_names.len() == elements_found
    }

    fn get_initializers(graph: GraphProto, node: NodeProto) -> Vec<TensorProto> {
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
}

#[allow(dead_code)]
pub struct OnnxModelEditor {}

impl OnnxModelEditor {
    #[allow(dead_code)]
    pub fn remove_node(node_name: String, model: ModelProto) -> ModelProto {
        let mut node_map: LinkedList<NodeProto> = LinkedList::new();
        for node in model.clone().graph.unwrap().node {
            // inserisco nella mappa nodi con chiave nome
            if node.name == node_name {
                //non inserisco il nodo
            } else {
                node_map.push_back(node);
            }
        }
        // nella mappa ho i nodi corretti da inserire
        // devo ricreare il modello
        let graph = GraphProto {
            node: node_map.into_iter().collect(),
            name: model.clone().graph.unwrap().name,
            initializer: model.clone().graph.unwrap().initializer, // Aggiungere eventuali inizializzatori
            sparse_initializer: model.clone().graph.unwrap().sparse_initializer, // Aggiungere eventuali inizializzatori sparsi
            doc_string: model.clone().graph.unwrap().doc_string,
            input: model.clone().graph.unwrap().input, // Aggiungere eventuali informazioni sugli input
            output: model.clone().graph.unwrap().output, // Aggiungere eventuali informazioninformazioni sugli output
            value_info: model.clone().graph.unwrap().value_info, // Aggiungere eventuali informazioni sui valori
            quantization_annotation: model.clone().graph.unwrap().quantization_annotation, // Aggiungere eventuali annotazioni di quantizzazione
        };
        let model_new = ModelProto {
            ir_version: model.ir_version,
            opset_import: model.opset_import, //opset_import,
            producer_name: model.producer_name,
            producer_version: model.producer_version,
            domain: model.domain,
            model_version: model.model_version,
            doc_string: model.doc_string,
            graph: Some(graph),
            metadata_props: model.metadata_props, // Aggiungere eventuali proprietà metadata
            training_info: model.training_info, // Aggiungere eventuali informazioni di addestramento
            functions: model.functions,         // Aggiungere eventuali funzioni locali
        };
        return model_new;
    }

    #[allow(dead_code)]
    pub fn insert_node(
        node_name: String,
        input: Vec<String>,
        output: Vec<String>,
        operation_type: String,
        domain: String,
        attribute: Vec<AttributeProto>,
        doc_string: String,
        model: &mut ModelProto,
    ) {
        let mut node_to_insert = NodeProto::new(input.clone(), output, node_name.clone(), operation_type, domain, attribute, doc_string);
    
        let nodes = &mut model.graph.as_mut().unwrap().node;

        //se non ci sono input il nodo viene inserito all'inizio del grafico
        if input.is_empty() {
            node_to_insert.input.push(nodes.get(0).unwrap().input.get(0).unwrap().to_string());
            nodes.insert(0, node_to_insert);
        }else{
            let mut node_to_insert_index = 0;
            for (index, node) in nodes.iter().enumerate(){
                if input.contains(&node.name) {
                    node_to_insert_index = index;
                }
            }
            nodes.insert(node_to_insert_index + 1, node_to_insert)
        }
    }

    #[allow(dead_code)]
    pub fn modify_node(
        node_name: String,
        input: Vec<String>,
        output: Vec<String>,
        operation_type: String,
        domain: String,
        attribute: Vec<AttributeProto>,
        doc_string: String,
        model: ModelProto,
    ) -> ModelProto {
        let mut node_map: LinkedList<&NodeProto> = LinkedList::new();
        let node_to_insert = NodeProto::new(
            input,
            output,
            node_name.clone(),
            operation_type,
            domain,
            attribute,
            doc_string,
        );
        for node in &model.graph.as_ref().unwrap().node {
            // inserisco nella mappa nodi con chiave nome
            if node.name == node_name {
                //nodo da modificare
                // inserisco nella lista il nodo modificato
                node_map.push_back(&node_to_insert);
            } else {
                node_map.push_back(node);
            }
        }

        let mut model_new = model.clone();
        let mut graph = model_new.graph.unwrap();
        graph.node = node_map.into_iter().map(|x| x.clone()).collect();
        model_new.graph = Some(graph);
        return model_new;
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

unsafe impl Sync for NodeIO {}

fn find_and_do_operation(
    node_for_op: &NodeProto,
    initializers: Vec<TensorProto>,
    inputs: Vec<TensorProto>,
    is_par_enabled: bool,
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
        "EXP" => exp(inputs, node_for_op, is_par_enabled),
        "CONCAT" => concat(&inputs, node_for_op), // use initializers ?
        "FLATTEN" => flatten(inputs, node_for_op),
        "RESHAPE" => reshape(inputs, initializers, node_for_op), //    input: ArrayViewD<f32>,shape: &Vec<isize>,allow_zero: i64,
        "CONV" => conv(inputs, initializers, node_for_op, is_par_enabled),
        "MAXPOOL" => maxpool(inputs, node_for_op, is_par_enabled),
        "AVERAGEPOOL" => globalavgpool(inputs, node_for_op, is_par_enabled),
        "BATCHNORMALIZATION" => batch_norm(inputs, initializers, node_for_op, is_par_enabled),
        "DROPOUT" => dropout(inputs, initializers, node_for_op),
        "SOFTMAX" => softmax(inputs, node_for_op, is_par_enabled),
        "GEMM" => gemm(inputs, initializers, node_for_op, is_par_enabled),
        "MATMUL" => matmul(inputs, initializers, node_for_op, is_par_enabled),
        "REDUCESUM" => reducesum(inputs, node_for_op),
        "GLOBALAVERAGEPOOL" => globalavgpool(inputs, node_for_op, is_par_enabled),
        "LRN" => lrn(inputs, node_for_op),
        _ => Err(OnnxError::UnsupportedOperation(
            "Operazione non supportata".to_string(),
        )),
    }
}
#[allow(dead_code)]
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

impl NodeProto {
    pub fn new(
        input: Vec<String>,
        output: Vec<String>,
        name: String,
        op_type: String,
        domain: String,
        attribute: Vec<AttributeProto>, // Assicurati di avere la definizione corretta di AttributeProto
        doc_string: String,
    ) -> Self {
        NodeProto {
            input,
            output,
            name,
            op_type,
            domain,
            attribute,
            doc_string,
        }
    }
}
impl ModelProto {
    #[allow(dead_code)]
    pub fn new(
        ir_version: i64,
        opset_import: Vec<OperatorSetIdProto>,
        producer_name: String,
        producer_version: String,
        domain: String,
        model_version: i64,
        doc_string: String,
        graph: Option<GraphProto>,
        metadata_props: Vec<StringStringEntryProto>,
        training_info: Vec<TrainingInfoProto>,
        functions: Vec<FunctionProto>,
    ) -> Self {
        ModelProto {
            ir_version,
            opset_import,
            producer_name,
            producer_version,
            domain,
            model_version,
            doc_string,
            graph,
            metadata_props,
            training_info,
            functions,
        }
    }
}
impl GraphProto {
    #[allow(dead_code)]
    pub fn new(
        node: Vec<NodeProto>,
        name: String,
        initializer: Vec<TensorProto>,
        sparse_initializer: Vec<SparseTensorProto>,
        doc_string: String,
        input: Vec<ValueInfoProto>,
        output: Vec<ValueInfoProto>,
        value_info: Vec<ValueInfoProto>,
        quantization_annotation: Vec<TensorAnnotation>,
    ) -> Self {
        GraphProto {
            node,
            name,
            initializer,
            sparse_initializer,
            doc_string,
            input,
            output,
            value_info,
            quantization_annotation,
        }
    }
}
