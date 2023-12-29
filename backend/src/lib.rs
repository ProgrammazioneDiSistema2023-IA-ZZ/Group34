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
