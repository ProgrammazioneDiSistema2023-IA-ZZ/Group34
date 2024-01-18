pub mod stateful_backend_environment;
pub mod onnx;
pub mod onnx_running_environment;
pub mod operations;
pub mod utils;
#[allow(unused_imports)]
use crate::utils::OnnxError;
mod js_binding {
    use neon::prelude::*;
    #[allow(unused_imports)]
    use prost::Message;
    use serde::Deserialize;
    #[allow(unused_imports)]
    use serde::Serialize;
    #[allow(unused_imports)]
    use serde_json::json;
    use crate::onnx::GraphProto;
    use crate::stateful_backend_environment::{self, NodeDto};

    fn hello(mut cx: FunctionContext) -> JsResult<JsString> {
        Ok(cx.string("hello node!!"))
    }
    fn start(mut cx: FunctionContext) -> JsResult<JsBoolean> {
        match stateful_backend_environment::start() {
            Ok(_) => Ok(cx.boolean(true)),
            Err(e) => {
                println!("{}", e);
                Ok(cx.boolean(false))
            }
        }
        //Ok(cx.boolean(stateful_backend_environment::start().is_ok()))
    }

    #[derive(Serialize)]
    struct JsonNode {
        id: u32,
        label: String,
    }

    #[derive(Serialize)]
    struct JsonEdge {
        from: u32,
        to: u32,
    }

    #[derive(Serialize)]
    struct JsonResult {
        nodes: Vec<JsonNode>,
        edges: Vec<JsonEdge>,
        initializers: Vec<String>,
    }

    impl From<&GraphProto> for JsonResult {
        fn from(graph_proto: &GraphProto) -> Self {
            let mut node_id_counter = 1;
            let mut node_id_map = std::collections::HashMap::new();
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut initializers = Vec::new();

            // Convert nodes
            for node_proto in &graph_proto.node {
                let label = node_proto.name.clone();
                let id = node_id_counter;
                node_id_map.insert(node_proto.name.clone(), id);
                node_id_counter += 1;
                nodes.push(JsonNode { id, label });
            }

            // Convert initializers
            for tensor_proto in &graph_proto.initializer {
                initializers.push(tensor_proto.name.clone());
            }

            // Convert edges
            for node_proto in &graph_proto.node {
                for input in &node_proto.input {
                    if let Some(&from_id) = node_id_map.get(&node_proto.name) {
                        if let Some(&to_id) = node_id_map.get(input) {
                            edges.push(JsonEdge { from: to_id , to: from_id });
                        }
                    }
                }
            }

            JsonResult { nodes, edges, initializers }
        }
    }

    fn select_model(mut cx: FunctionContext) -> JsResult<JsString> {
        let n_model = cx.argument::<JsNumber>(0)?.value(&mut cx);
        let graph = stateful_backend_environment::select_model(n_model as usize).graph.unwrap();
        // Convert GraphProto to JSON structure
        let json_result: JsonResult = (&graph).into();

        // Convert the JsonResult to JSON string
        let json_string = serde_json::to_string(&json_result).expect("Failed to convert to JSON string");
        /*
        let a = stateful_backend_environment::select_model(1).graph.unwrap().as_bytes();
        Ok(cx.string::<String>( Message::decode(&a[..]).unwrap()))

         */

        /*
        // Convert GraphProto to a byte vector
        let mut buf = Vec::new();
        stateful_backend_environment::select_model(1).graph.unwrap().encode(&mut buf).expect("Failed to encode GraphProto");

        // Convert byte vector to a JSON value
        let json_value = json!({
        "graph_proto": base64::encode(&buf), // Assuming you want to encode the binary data
        // Add other fields as needed
    });

        // Convert JSON value to a JSON string
        let json_string = serde_json::to_string_pretty(&json_value).expect("Failed to convert to JSON string");
        Ok(cx.string::<String>( json_string))

         */
        Ok(cx.string::<String>(json_string))

    }

    
    #[derive(Debug, Deserialize)]
    struct Options {
        use_default: bool,
        image: i32,
        use_parallelization: bool,
    }

    #[derive(Debug, Deserialize)]
    struct RequestBody {
        options: Options,
    }

    fn run(mut cx: FunctionContext) -> JsResult<JsString>{

        
        let json_string = cx.argument::<JsString>(0)?.value(&mut cx);
    
        // Deserializza l'oggetto JSON in una struct
        let request_body: RequestBody = serde_json::from_str(&json_string).expect("Failed to convert to JSON string");
        let use_default = request_body.options.use_default;
        let image = request_body.options.image;
        let use_parallelization = request_body.options.use_parallelization;
        let mut path_img="";
        if use_default==true {
            if use_parallelization==true {
                let result = stateful_backend_environment::run(true,false,"".to_string());
                Ok(cx.string(result))
            }else{
                let result = stateful_backend_environment::run(false,false,"".to_string());
                Ok(cx.string(result))
            }
        }else{
            match image {
                0 => {
                    path_img="./src/images/ape.jpg";
                }
                1 => {
                    path_img="./src/images/aquila.jpg";
                }
                2 => {
                    path_img="./src/images/gatto.jpg";
                }
                _ => {
                    println!("Image is not 0, 1, or 2");
                }
            }
            if use_parallelization==true {
                let result = stateful_backend_environment::run(true,true,path_img.to_string());
                Ok(cx.string(result))
            }else{
                let result = stateful_backend_environment::run(false,true,path_img.to_string());
                Ok(cx.string(result))
            }
        }
    }

    fn get_node_js(mut cx: FunctionContext) -> JsResult<JsString>{
        let node_dto = stateful_backend_environment::get_node(cx.argument::<JsString>(0)?.value(&mut cx));
        let result = serde_json::to_string(&node_dto).expect("Failed to serialize to JSON");
        Ok(cx.string(result))
    }

    fn create_node(mut cx: FunctionContext) -> JsResult<JsString>{
        let node:NodeDto = serde_json::from_str(&cx.argument::<JsString>(0)?.value(&mut cx)).expect("Failed to serialize to JSON");
        let graph = stateful_backend_environment::create_node(node).graph.unwrap();
        // Convert GraphProto to JSON structure
        let json_result: JsonResult = (&graph).into();

        // Convert the JsonResult to JSON string
        let json_string = serde_json::to_string(&json_result).expect("Failed to convert to JSON string");
        Ok(cx.string::<String>(json_string))
    }

    fn modify_node(mut cx: FunctionContext) -> JsResult<JsString>{
        let node:NodeDto = serde_json::from_str(&cx.argument::<JsString>(0)?.value(&mut cx)).expect("Failed to serialize to JSON");
        let graph = stateful_backend_environment::modify_node(node).graph.unwrap();
        // Convert GraphProto to JSON structure
        let json_result: JsonResult = (&graph).into();

        // Convert the JsonResult to JSON string
        let json_string = serde_json::to_string(&json_result).expect("Failed to convert to JSON string");
        Ok(cx.string::<String>(json_string))
    }

    fn delete_node(mut cx: FunctionContext) -> JsResult<JsString>{
        let node_name = cx.argument::<JsString>(0)?.value(&mut cx);
        let graph = stateful_backend_environment::remove_node(node_name).graph.unwrap();
        // Convert GraphProto to JSON structure
        let json_result: JsonResult = (&graph).into();

        // Convert the JsonResult to JSON string
        let json_string = serde_json::to_string(&json_result).expect("Failed to convert to JSON string");
        Ok(cx.string::<String>(json_string))
    }

    #[neon::main]
    fn main_js(mut cx: ModuleContext) -> NeonResult<()> {
        cx.export_function("hello", hello)?;
        cx.export_function("start", start)?;
        cx.export_function("select_model", select_model)?;
        cx.export_function("run", run)?;
        cx.export_function("get_node_js", get_node_js)?;
        cx.export_function("create_node", create_node)?;
        cx.export_function("modify_node", modify_node)?;
        cx.export_function("delete_node", delete_node)?;
        Ok(())
    }
}

#[cfg(feature = "include_pyo3")]
mod python_binding {
    use pyo3::prelude::*;

    use std::io::{self, Write};
    use pyo3::prelude::*;
    use tract_onnx::tract_core::tract_data::itertools::Itertools;
    use crate::{utils::{get_path_from_ordinal, CLASSES_NAMES, decode_message, convert_img}, onnx::{TensorProto, ModelProto}, operations::utils::{tensor_proto_to_ndarray, ndarray_to_tensor_proto}, onnx_running_environment::OnnxRunningEnvironment};

    #[pyfunction]
    fn main_python() {
        main();
        //todo!("Gioele metti main");
        fn get_bool_from_console(prompt: &str) -> bool {
            loop {
                io::stdout().flush().unwrap();
                println!("{prompt}");
                let mut choice: String = String::new();
                io::stdin()
                    .read_line(&mut choice)
                    .expect("Errore durante la lettura dell'input");
                // Rimuovi spazi e caratteri di nuova linea dall'input
                let choice = choice.trim();
                match choice {
                    "s" => return true,
                    "n" => return false,
                    _ => println!("Scelta non valida. Riprova."),
                }
            }
        }

        fn get_int_from_console(prompt: &str, min: i32, max: i32) -> i32 {
            loop {
                io::stdout().flush().unwrap();
                println!("{prompt}");
                let mut choice: String = String::new();
                io::stdin()
                    .read_line(&mut choice)
                    .expect("Errore durante la lettura dell'input");
                // Rimuovi spazi e caratteri di nuova linea dall'input
                let choice = choice.trim();
                let value: i32 = choice.to_string().parse().unwrap_or(min - 1);
                if value < min || value > max {
                    println!("Scelta non valida. Riprova.")
                } else {
                    return value;
                };
            }
        }

        fn get_string_from_console(prompt: &str) -> String {
            io::stdout().flush().unwrap();
            println!("{prompt}");
            let mut choice: String = String::new();
            io::stdin()
                .read_line(&mut choice)
                .expect("Errore durante la lettura dell'input");
            // Rimuovi spazi e caratteri di nuova linea dall'input
            let choice = choice.trim();
            choice.to_string()
        }

        fn main() {
            loop {
                println!("
            ————————————————————————————————————————————————————
                                    ONNX
            ————————————————————————————————————————————————————
            ");

                let is_run_par_enabled = get_bool_from_console("\nvuoi eseguire la rete in modo parallelo? (s/n) \n indicando 'n' sarà eseguita in modo sequenziale");

                let model_index = get_int_from_console(
                    "scegli una rete:
            1. mobilenet
            2. resnet
            3. squeezenet
            4. caffenet
            5. alexnet
            6. fine",
                    1,
                    6,
                );

                let path = get_path_from_ordinal(model_index as usize);
                if path.is_none() {
                    print!("A presto :)");
                    return;
                };
                let path = path.unwrap();

                let is_def_test_set = get_bool_from_console("\nvuoi usare il test set di default ? (s/n)");

                let mut use_custom_img = false;
                let mut path_img = "".to_string();
                if !is_def_test_set {
                    path_img = get_string_from_console("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/images/nomeimmagine.formato");
                    use_custom_img = true
                }

                // uso immagine fornita da utente
                let model_proto: ModelProto = decode_message(&path.model);

                println!("Reading the inputs ...");
                let mut input_tensor: TensorProto = decode_message(&path.test);
                // uso immagine
                if use_custom_img {
                    let arrD_img = convert_img(path_img.to_string());
                    input_tensor = ndarray_to_tensor_proto::<f32>(arrD_img, "data").unwrap();
                }

                println!("starting Network...");
                let new_env = OnnxRunningEnvironment::new(model_proto, input_tensor);

                if is_run_par_enabled {
                    let pred_out = new_env.run(is_run_par_enabled); //predicted output par
                    println!("Predicted classes:");
                    print_results(pred_out);
                } else {
                    let pred_out = new_env.run_sequential(is_run_par_enabled); //predicted output seq
                    println!("Predicted classes:");
                    print_results(pred_out);
                }

                if !use_custom_img {
                    let output_tensor: TensorProto = decode_message(&path.output);
                    println!("\nGround truth classes:");
                    print_results(output_tensor);
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
                print!(
                    "|Class n:{} Value:{}| ",
                    CLASSES_NAMES[element.0], element.1
                );
            }
        }
    }

    #[pymodule]
    fn group_34(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(main_python, m)?)?;
        Ok(())
    }
}
