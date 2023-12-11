#[allow(unused_imports)]
mod onnx {
    include!("onnx.rs");
}
use crate::onnx::*;
use crate::onnx::NodeProto;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;
use crate::onnx_running_environment::OnnxRunningEnvironment;
use crate::operations::utils::ndarray_to_tensor_proto;

use crate::utils::CLASSES_NAMES;


use std::collections::LinkedList;
use image::GenericImage;
use image::imageops;
use ndarray::Array3;
use ndarray::{Array, Array2, Array4, ArrayD, Axis, Dimension, IxDyn};
use onnx::AttributeProto;
use onnx::FunctionProto;
use onnx::GraphProto;
use onnx::OperatorSetIdProto;
use onnx::StringStringEntryProto;
use onnx::TrainingInfoProto;
use onnx::tensor_proto::DataLocation;
use onnx::ModelProto;
use operations::utils::tensor_proto_to_ndarray;
use rand::prelude::*;
use image::{GenericImageView, DynamicImage};
use tract_onnx::tract_core::tract_data::itertools::Itertools;
use std::any::type_name;

use std::fs::File;
use std::io::{self, Read, Write};

use std::process::exit;
mod onnx_running_environment;
mod operations;
mod utils;

const MIN: u32 = 256;
const CROP: u32 = 224;
// valori standard utilizzati
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const SCALEFACTOR: f32 = 255.0;
fn main() {
    let mut path_model: &str = "";
    let mut path_testset: &str = "";
    let mut path_output: &str = "";
    let mut use_custom_img=0;
    let mut path_img="";
    let mut flag_execution=true;
    loop {
        io::stdout().flush().unwrap();
        println!("\nvuoi usare eseguire la sete in modo parallelo? (s/n) \n indicando 'n' sarà eseguita in modo sequenziale");
        let mut choice4 = String::new();
        io::stdin()
            .read_line(&mut choice4)
            .expect("Errore durante la lettura dell'input");
        // Rimuovi spazi e caratteri di nuova linea dall'input
        let choice4 = choice4.trim();
        match choice4 {
            "s" => {
                flag_execution=true;
                break;
            }
            "n" => {
                flag_execution=false;
                break;
            }
            _ => println!("Scelta non valida. Riprova."),
        }
    }
    let mut take_choice = String::new();
    loop {
        println!("Onnx runtime");
        println!("scegli una rete:");
        println!("1. mobilenet");
        println!("2. resnet");
        println!("3. squeezenet");
        println!("4. caffenet");
        println!("5. alexnet");
        println!("6. fine");

        print!("Seleziona un'opzione: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin()
            .read_line(&mut choice)
            .expect("Errore durante la lettura dell'input");

        // Rimuovi spazi e caratteri di nuova linea dall'input
        let choice = choice.trim();
        match choice {
            "1" => {
                path_model = &mobilenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &mobilenet_load_testset();
                            path_output = &mobilenet_load_output();
                            break;
                        }
                        "n" => {
                            println!("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/nomeimmagine");
                            io::stdin()
                                .read_line(&mut take_choice)
                                .expect("Errore durante la lettura dell'input");
                            // Rimuovi spazi e caratteri di nuova linea dall'input
                            path_img = take_choice.trim();
                            //prepar per fare conversione
                            use_custom_img=1;
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "2" => {
                path_model = &resnet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &resnet_load_testset();
                            path_output = &resnet_load_output();
                            break;
                        }
                        "n" => {
                            println!("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/nomeimmagine");
                            io::stdin()
                                .read_line(&mut take_choice)
                                .expect("Errore durante la lettura dell'input");
                            // Rimuovi spazi e caratteri di nuova linea dall'input
                            path_img = take_choice.trim();
                            //prepar per fare conversione
                            use_custom_img=1;
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "3" => {
                path_model = &squeezenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &squeezenet_load_testset();
                            path_output = &squeezenet_load_output();
                            break;
                        }
                        "n" => {
                            println!("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/nomeimmagine");
                            io::stdin()
                                .read_line(&mut take_choice)
                                .expect("Errore durante la lettura dell'input");
                            // Rimuovi spazi e caratteri di nuova linea dall'input
                            path_img = take_choice.trim();
                            //prepar per fare conversione
                            use_custom_img=1;
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "4" => {
                path_model = &caffenet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &caffenet_load_testset();
                            path_output = &caffenet_load_output();
                            break;
                        }
                        "n" => {
                            println!("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/nomeimmagine");
                            io::stdin()
                                .read_line(&mut take_choice)
                                .expect("Errore durante la lettura dell'input");
                            // Rimuovi spazi e caratteri di nuova linea dall'input
                            path_img = take_choice.trim();
                            //prepar per fare conversione
                            use_custom_img=1;
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "5" => {
                path_model = &alexnet_load();
                loop {
                    io::stdout().flush().unwrap();
                    println!("vuoi usare il test set di default ? (s/n)");
                    let mut choice2 = String::new();
                    io::stdin()
                        .read_line(&mut choice2)
                        .expect("Errore durante la lettura dell'input");
                    // Rimuovi spazi e caratteri di nuova linea dall'input
                    let choice2 = choice2.trim();
                    match choice2 {
                        "s" => {
                            path_testset = &alexnet_load_testset();
                            path_output = &alexnet_load_output();
                            break;
                        }
                        "n" => {
                            println!("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/nomeimmagine");
                            io::stdin()
                                .read_line(&mut take_choice)
                                .expect("Errore durante la lettura dell'input");
                            // Rimuovi spazi e caratteri di nuova linea dall'input
                            path_img = take_choice.trim();
                            //prepar per fare conversione
                            use_custom_img=1;
                            break;
                        }
                        _ => println!("Scelta non valida. Riprova."),
                    }
                }
                break;
            }
            "6" => {
                println!("Uscita dal programma");
                break;
            }
            _ => println!("Scelta non valida. Riprova."),
        }
    }
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    //let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    if (path_model.is_empty() || path_testset.is_empty()) && use_custom_img==0 {
        exit(1)
    }
    if (path_model.is_empty() || path_testset.is_empty()) && use_custom_img==1 {
        // uso immagine fornita da utente
        let data = std::fs::read(path_model).expect("Failed to read ProtoBuf file");
        let model_proto: ModelProto =
            prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

        println!("Reading the inputs ...");
        // uso immagine 
        let arrD_img= convert_img(path_img.to_string());
        let input_tensor: TensorProto =ndarray_to_tensor_proto::<f32>(arrD_img, "data").unwrap();

        println!("starting Network...");
        let new_env = OnnxRunningEnvironment::new(model_proto, input_tensor);

        if flag_execution==true {
            let pred_out = new_env.run(); //predicted output par
            println!("Predicted classes:");
            print_results(pred_out);
        }else{
            let pred_out = new_env.run_sequential(); //predicted output seq
            println!("Predicted classes:");
            print_results(pred_out);
        }

    }
    if use_custom_img==0 {
        let data = std::fs::read(path_model).expect("Failed to read ProtoBuf file");
        let model_proto: ModelProto =
            prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

        println!("Reading the inputs ...");
        let data = std::fs::read(path_testset).expect("Failed to read ProtoBuf file");
        let input_tensor: TensorProto =
            prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

        println!("starting Network...");
        let new_env = OnnxRunningEnvironment::new(model_proto.clone(), input_tensor);
        if flag_execution==true {
            let pred_out = new_env.run(); //predicted output par
            println!("Predicted classes:");
            print_results(pred_out);
        }else{
            let pred_out = new_env.run_sequential(); //predicted output seq
            println!("Predicted classes:");
            print_results(pred_out);
        }
        let data = std::fs::read(path_output).expect("Failed to read ProtoBuf file");
        let output_tensor: TensorProto =
            prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

        println!("\nGround truth classes:");
        print_results(output_tensor);
    }
}
fn remove_node(node_name:String,model:ModelProto)-> ModelProto{
    let mut node_map: LinkedList<NodeProto> = LinkedList::new();
    for node in model.clone().graph.unwrap().node {
        // inserisco nella mappa nodi con chiave nome
        if node.name==node_name {
            //non inserisco il nodo
        }else{
            node_map.push_back(node);
        }
    }
    // nella mappa ho i nodi corretti da inserire 
    println!("{:?}", node_map);
    // devo ricreare il modello
    let graph=GraphProto {
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
        functions: model.functions, // Aggiungere eventuali funzioni locali
    };
    return model_new
}

fn insert_node(node_name:String,
    model:ModelProto,
    input: Vec<String>,
    _initializers: Vec<String>,
    output:Vec<String>,
    operation_type:String,
    domain:String,
    attribute: Vec<AttributeProto>,
    doc_string: String,
    node_before: Option<NodeProto>,// uso questi parametri per inserire il nodo in una pos specifica
    node_after: Option<NodeProto>// possono essere entrambi none se per esempio è il primo nodo
)-> ModelProto{
    let mut node_map: LinkedList<NodeProto> = LinkedList::new();
    let node_to_insert = NodeProto::new(
        input,
        output,
        node_name.clone(),
        operation_type,
        domain,
        attribute, 
        doc_string,
    );
    let mut before: Option<NodeProto>=None;
    for node in model.clone().graph.unwrap().node {

        if node.name==node_after.clone().unwrap().name || node.name==node_before.clone().unwrap().name
         {
            //controllo se il nome del nodo successivo è uguale
            if before.clone().unwrap().name==node_before.clone().unwrap().name || before.clone().is_none()==node_before.clone().is_none() {
                // se anche il nome del nodo precedente corrisponde
                // inserisco il nuovo nodo nella posizione specificata
                node_map.push_back(node_to_insert.clone());
            }
    
        
        }else{
            node_map.push_back(node.clone());
        }
        // salvo il nodo precedente
        before = Some(node.clone());
    }
    // se il nodo che voglio inserire è l'ultimo non avra un successivo quindi ho salvato 
    // uscendo dal for before come ultimo nodo della rete quindi vado a fare push 
    if before.clone().unwrap().name==node_before.clone().unwrap().name && node_before.clone().is_none() {
        // se anche il nome del nodo precedente corrisponde
        // inserisco il nuovo nodo nella posizione specificata
        // in questo caso sarà l'ultimo nodo 
        node_map.push_back(node_to_insert.clone());
    }
    let graph=GraphProto {
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
        opset_import: model.opset_import,
        producer_name: model.producer_name,
        producer_version: model.producer_version,
        domain: model.domain,
        model_version: model.model_version,
        doc_string: model.doc_string,
        graph: Some(graph),
        metadata_props: model.metadata_props, 
        training_info: model.training_info, 
        functions: model.functions, 
    };
    return model_new;
}
fn modify_node(node_name:String,
    model:ModelProto,
    input: Vec<String>,
    _initializers: Vec<String>,
    output:Vec<String>,
    operation_type:String,
    domain:String,
    attribute: Vec<AttributeProto>,
    doc_string: String
)-> ModelProto{
    let mut node_map: LinkedList<NodeProto> = LinkedList::new();
    let node_to_insert = NodeProto::new(
        input,
        output,
        node_name.clone(),
        operation_type,
        domain,
        attribute, 
        doc_string,
    );
    for node in model.clone().graph.unwrap().node {
        // inserisco nella mappa nodi con chiave nome
        if node.name==node_name {
            //nodo da modificare 
            // inserisco nella lista il nodo modificato
            node_map.push_back(node_to_insert.clone());
        
        }else{
            node_map.push_back(node);
        }
    }
    let graph=GraphProto {
        node: node_map.into_iter().collect(),
        name: model.clone().graph.unwrap().name,
        initializer: model.clone().graph.unwrap().initializer, 
        sparse_initializer: model.clone().graph.unwrap().sparse_initializer, 
        doc_string: model.clone().graph.unwrap().doc_string,
        input: model.clone().graph.unwrap().input, 
        output: model.clone().graph.unwrap().output, 
        value_info: model.clone().graph.unwrap().value_info, 
        quantization_annotation: model.clone().graph.unwrap().quantization_annotation, 
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
        functions: model.functions, // Aggiungere eventuali funzioni locali
    };
    return model_new;
}
pub fn new_node_proto() -> NodeProto {
    NodeProto {
        input: Vec::new(),
        output: Vec::new(),
        name: String::new(),
        op_type: String::new(),
        domain: String::new(),
        attribute: Vec::new(), // Assicurati di avere la definizione corretta di AttributeProto
        doc_string: String::new(),
    }
}
impl NodeProto {
    pub fn new(
        input: Vec<String>,
        output: Vec<String>,
        name: String,
        op_type: String,
        domain: String,
        attribute: Vec<AttributeProto>,  // Assicurati di avere la definizione corretta di AttributeProto
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
fn print_results(tensor: TensorProto) {
    let data = tensor_proto_to_ndarray::<f32>(&tensor).unwrap();

    for element in data
        .iter()
        .enumerate()
        .sorted_by(|a, b| b.1.total_cmp(a.1))
        .take(3)
    {
        print!("|Class n:{} Value:{}| ", CLASSES_NAMES[element.0], element.1);
    }
}

fn read_input(input: &str) {
    // Path to your .pb file da concatenare
    let file_path = input;

    // Open the file
    let mut file = File::open(file_path).expect("Unable to open file");

    // Read the file contents into a Vec<u8>
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Unable to read file");

    // Deserialize the .pb file
    //let mut message = your_proto::YourMessage::new();  // Replace with your generated protobuf message type
    //message.merge_from_bytes(&buffer).expect("Unable to parse .pb file");

    // Access the data in the protobuf message
    //println!("Field value: {:?}", message.get_your_field());

    // If not preprocessing, proceed with model loading and inference
}

fn mobilenet_load_testset() -> &'static str {
    let path_testset = "src/models/mobilenet/data_mobilenet/input_0.pb";
    return path_testset;
}

fn mobilenet_load() -> &'static str {
    let path_model = "src/models/mobilenet/model.onnx";
    return path_model;
}

fn mobilenet_load_output() -> &'static str {
    let path_output = "src/models/mobilenet/data_mobilenet/output_0.pb";
    return path_output;
}

fn googlenet_load_testset() -> &'static str {
    let path_testset = "src/models/googlenet/data_googlenet/input_0.pb";
    return path_testset;
}

fn googlenet_load() -> &'static str {
    let path_model = "src/models/googlenet/model.onnx";
    return path_model;
}
fn googlenet_load_output() -> &'static str {
    let path_output = "src/models/googlenet/data_googlenet/output_0.pb";
    return path_output;
}
fn resnet_load_testset() -> &'static str {
    let path_testset = "src/models/resnet/data_resnet/input_0.pb";
    return path_testset;
}

fn resnet_load() -> &'static str {
    let path_model = "src/models/resnet/model.onnx";
    return path_model;
}

fn resnet_load_output() -> &'static str {
    let path_output = "src/models/resnet/data_resnet/output_0.pb";
    return path_output;
}

fn squeezenet_load_testset() -> &'static str {
    let path_testset = "src/models/squeezenet/data_squeezenet/input_0.pb";
    return path_testset;
}

fn squeezenet_load() -> &'static str {
    let path_model = "src/models/squeezenet/model.onnx";
    return path_model;
}

fn squeezenet_load_output() -> &'static str {
    let path_output = "src/models/squeezenet/data_squeezenet/output_0.pb";
    return path_output;
}


fn alexnet_load_testset() -> &'static str {
    let path_testset = "src/models/alexnet/data_alexnet/input_0.pb";
    return path_testset;
}

fn alexnet_load() -> &'static str {
    let path_model = "src/models/alexnet/model.onnx";
    return path_model;
}

fn alexnet_load_output() -> &'static str {
    let path_output = "src/models/alexnet/data_alexnet/output_0.pb";
    return path_output;
}


fn caffenet_load_testset() -> &'static str {
    let path_testset = "src/models/caffenet/data_caffenet/input_0.pb";
    return path_testset;
}

fn caffenet_load() -> &'static str {
    let path_model = "src/models/caffenet/model.onnx";
    return path_model;
}
fn caffenet_load_output() -> &'static str {
    let path_output = "src/models/caffenet/data_caffenet/output_0.pb";
    return path_output;
}
fn resize_image(image: DynamicImage, width: u32, height: u32, new_width: u32, new_height: u32) -> DynamicImage {
    // Resize the image
    let resized_image = image.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);

    // Create a new image with the desired dimensions (1*3*224*224)
    let mut final_image = DynamicImage::new_rgba8(width * new_width, height * new_height);

    // Paste the resized image into the final image
    final_image.copy_from(&resized_image, 0, 0);

    final_image
}
fn convert_img(path: String) -> ArrayD<f32> {
    // Load the image
    let mut img = image::open(path).unwrap();

    let (width, height) = img.dimensions();

    // faccio il resize a min 
    let (nwidth, nheight) = if width > height {
        (MIN* width / height, MIN)
    } else {
        (MIN,MIN* height / width)
    };

    img = img.resize(nwidth, nheight, imageops::FilterType::Gaussian);

    // Crop the image to CROP_SIZE from the center
    let crop_x = (nwidth - CROP) / 2;
    let crop_y = (nheight - CROP) / 2;

    img = img.crop_imm(crop_x, crop_y, CROP, CROP); // faccio crop a 224 come voglio input

    //trasformo in rgb e poi trasformo in ndarray per utilizzarlo
    let img_rgb = img.to_rgb8();

    let raw_data = img_rgb.into_raw();

    let (mut rs, mut gs, mut bs) = (Vec::new(), Vec::new(), Vec::new());

    for i in 0..raw_data.len() / 3 {
        rs.push(raw_data[3 * i]);
        gs.push(raw_data[3 * i + 1]);
        bs.push(raw_data[3 * i + 2]);
    }

    let r_arr: Array2<u8> =Array::from_shape_vec((CROP as usize, CROP as usize), rs).unwrap();
    let g_arr: Array2<u8> =Array::from_shape_vec((CROP as usize, CROP as usize), gs).unwrap();
    let b_arr: Array2<u8> =Array::from_shape_vec((CROP as usize, CROP as usize), bs).unwrap();
    //creo Array3
    let mut arr_fin: Array3<u8> =ndarray::stack(Axis(2), &[r_arr.view(), g_arr.view(), b_arr.view()]).unwrap();
    //faccio trasposta
    arr_fin.swap_axes(0, 2);

    let mean = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            MEAN[0] * SCALEFACTOR,
            MEAN[1] * SCALEFACTOR,
            MEAN[2] * SCALEFACTOR,
        ],
    )
    .unwrap();

    let std = Array::from_shape_vec(
        (3, 1, 1),
        vec![
            STD[0] * SCALEFACTOR,
            STD[1] * SCALEFACTOR,
            STD[2] * SCALEFACTOR,
        ],
    )
    .unwrap();

    let mut arr_f: Array3<f32> = arr_fin.mapv(|x| x as f32);

    arr_f -= &mean;
    arr_f /= &std;

    //aggiungo la batch dim
    let arr_f_batch: Array4<f32> = arr_f.insert_axis(Axis(0));

    // Convert Array4 to ArrayD
    let arr_d: ArrayD<f32> = arr_f_batch.into_dimensionality().unwrap();

    arr_d
}

struct Operation {
    op_type: OperationType,
    input: Vec<TensorProto>,
    op_attributes: Vec<AttributeProto>,
}

fn from<T>(array: ArrayD<T>, name: String) -> Result<TensorProto, OnnxError>
where
    T: Into<f32> + Into<f64> + Into<i32> + Into<i64>,
{
    let mut tensor = TensorProto {
        dims: array.shape().iter().map(|&x| x as i64).collect(),
        data_type: DataType::Undefined.into(),
        segment: None,
        name: name,
        doc_string: "".to_string(),
        data_location: DataLocation::Default.into(),
        float_data: Vec::new(),
        int32_data: Vec::new(),
        string_data: Vec::new(),
        int64_data: Vec::new(),
        raw_data: Vec::new(),
        external_data: Vec::new(),
        double_data: Vec::new(),
        uint64_data: Vec::new(),
    };
    match type_name::<T>() {
        "f32" => {
            tensor.data_type = DataType::Float.into();
            tensor.float_data = array.into_raw_vec().into_iter().map(|x| x.into()).collect();
            Ok(tensor)
        }
        _ => Err(OnnxError::new("Unsupported data type")),
    }
}

#[derive(Debug)]
pub enum OnnxError {
    /// Indicates that a required attribute was not found.
    ///
    /// The contained `String` provides the name or identifier of the missing attribute.
    AttributeNotFound(String),

    /// Represents generic internal errors that might occur during processing.
    ///
    /// The contained `String` provides a description or message detailing the nature of the internal error.
    InternalError(String),

    /// Indicates an error that occurred during data type conversion.
    ///
    /// The contained `String` provides additional information about the conversion that failed.
    ConversionError(String),

    /// Represents an error where an operation or functionality is not supported.
    ///
    /// The contained `String` provides details about the unsupported operation.
    UnsupportedOperation(String),

    /// Indicates a mismatch between expected and actual tensor shapes.
    ///
    /// The contained `String` provides details about the shape mismatch, such as the expected vs. actual dimensions.
    ShapeMismatch(String),

    /// Represents an error where an expected input tensor or data is missing.
    ///
    /// The contained `String` provides details about the missing input.
    MissingInput(String),

    /// Indicates an error due to invalid data or values.
    ///
    /// The contained `String` provides details about the nature of the invalid data.
    InvalidValue(String),

    /// Indicates an error related to tensor shape computations.
    ///
    /// The contained `String` provides details about the shape computation error.
    ShapeError(String),
}

impl OnnxError {
    fn new(message: &str) -> OnnxError {
        OnnxError::InternalError(message.to_string())
    }
}

// impl fmt::Display for OnnxError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.message)
//     }
// }

// impl Error for OnnxError {
//     fn description(&self) -> &str {
//         &self.message
//     }
// }
impl TensorProto {
    pub fn new() -> TensorProto {
        ::std::default::Default::default()
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

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

fn into<T>(tensor: TensorProto) -> Result<ArrayD<T>, std::io::Error>
where
    T: From<f32>,
{
    let shape: Vec<usize> = tensor.dims.iter().map(|dim| *dim as usize).collect();

    match tensor.data_type {
        1 => {
            //let float_data: Result<Vec<T>, _> = tensor.float_data.into_iter().map(T::from).collect();
            let data: Vec<T> = tensor
                .float_data
                .iter()
                .map(|&value| value.into())
                .collect();
            Ok(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Unsupported data type",
        )),
    }
}
