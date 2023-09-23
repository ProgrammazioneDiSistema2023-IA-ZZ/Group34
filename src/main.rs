use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        // load the model
        // esempio squeeze
        .model_for_path("src/squeezenet.onnx")?
        // specify input type and shape
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?;
        // optimize the model
        //.into_optimized()?;
        // make the model runnable and fix its inputs and outputs

    model.nodes.iter().for_each(|x|{
        println!("{:?}",x);
    });

    // open image, resize it and make a Tensor out of it
    let image = image::open("src/image.jpg").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    })
        .into();
    let runModel = model.into_runnable()?;
    // run the model on the input
    let result = runModel.run(tvec!(image))?;

    println!("result: {:?}", result);
    Ok(())
}


