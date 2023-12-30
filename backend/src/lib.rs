mod js_binding {
    use neon::prelude::*;
    fn hello(mut cx: FunctionContext) -> JsResult<JsString> {
        Ok(cx.string("hello node!!"))
    }

    #[neon::main]
    fn main_js(mut cx: ModuleContext) -> NeonResult<()> {
        cx.export_function("hello", hello)?;
        Ok(())
    }
}

// mod python_binding {
//     use pyo3::prelude::*;

//     #[pyfunction]
//     fn main_python() {
//         todo!("Gioele metti main");
//     }

//     #[pymodule]
//     fn group_34(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
//         m.add_function(wrap_pyfunction!(main_python, m)?)?;
//         Ok(())
//     }
// }

mod onnx {
    include!("onnx.rs");
}

mod onnx_running_environment;
mod operations;
mod stateful_backend_environment;
mod utils;

mod python_binding {
    use pyo3::prelude::*;
    use super::*;
    use crate::onnx::ModelProto;
    use crate::onnx::TensorProto;
    use crate::onnx_running_environment::OnnxRunningEnvironment;
    use crate::operations::utils::ndarray_to_tensor_proto;

    use crate::utils::convert_img;
    use crate::utils::decode_message;
    use crate::utils::get_path_from_ordinal;
    use crate::utils::CLASSES_NAMES;
    use operations::utils::tensor_proto_to_ndarray;

    use std::io::{self, Read, Write};

    #[pymodule]
    fn group_34(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(main_python, m)?)?;
        Ok(())
    }

    //legge da console un valore di risposa a una s/n function
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

    #[pyfunction]
    fn main_python() {
        loop {
            println!(
                "
        ————————————————————————————————————————————————————
                                ONNX
        ————————————————————————————————————————————————————
        "
            );

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

            let is_def_test_set =
                get_bool_from_console("\nvuoi usare il test set di default ? (s/n)");

            let mut use_custom_img = false;
            let mut path_img = "".to_string();
            if !is_def_test_set {
                path_img = get_string_from_console("inserire il path dell'immagine che si vuole utilizzare\nl'immagine va inserita in src/nomeimmagine");
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
    /*
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
    */

    /*
    fn resize_image(image: DynamicImage, width: u32, height: u32, new_width: u32, new_height: u32) -> DynamicImage {
        // Resize the image
        let resized_image = image.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);

        // Create a new image with the desired dimensions (1*3*224*224)
        let mut final_image = DynamicImage::new_rgba8(width * new_width, height * new_height);

        // Paste the resized image into the final image
        final_image.copy_from(&resized_image, 0, 0);

        final_image
    }
    */
}
