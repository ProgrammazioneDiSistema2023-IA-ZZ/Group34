use std::path::Path;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    
    let onnx_path_file = Path::new("C:/Users/utente/Downloads/mobilenetv2-10.onnx");

    // Apri il file 
    let model = tract_onnx::onnx().model_for_path(onnx_path_file)?;

    // Stampa le informazioni sul modello
    println!("modello onnx : {:?}", model);

    Ok(())
}




