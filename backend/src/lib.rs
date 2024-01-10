pub mod stateful_backend_environment;
pub mod onnx;
pub mod onnx_running_environment;
pub mod operations;
pub mod utils;
use crate::utils::OnnxError;
mod js_binding {
    use neon::prelude::*;
    use prost::Message;
    use serde_json::json;
    use crate::stateful_backend_environment;

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

    fn select_model(mut cx: FunctionContext) -> JsResult<JsString> {
        /*
        let a = stateful_backend_environment::select_model(1).graph.unwrap().as_bytes();
        Ok(cx.string::<String>( Message::decode(&a[..]).unwrap()))

         */

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

    }

    #[neon::main]
    fn main_js(mut cx: ModuleContext) -> NeonResult<()> {
        cx.export_function("hello", hello)?;
        cx.export_function("start", start)?;
        cx.export_function("select_model", select_model)?;
        Ok(())
    }
}

#[cfg(feature = "include_pyo3")]
mod python_binding {
    use pyo3::prelude::*;

    #[pyfunction]
    fn main_python() {
        todo!("Gioele metti main");
    }

    #[pymodule]
    fn group_34(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(main_python, m)?)?;
        Ok(())
    }
}
