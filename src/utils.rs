use rand::{Rng, thread_rng};
use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;

pub fn get_random_float_tensor(dims: Vec<i64>) -> TensorProto{

    // Generate random data
    let mut rng = thread_rng();
    let data: Vec<f32> = (0..dims.iter().product::<i64>())
        .map(|_| rng.gen_range(0.0..1.0)) // Generate random floats between 0 and 1
        .collect();

    // Create a new TensorProto and set its properties
    let mut tensor_proto = TensorProto::default();
    tensor_proto.dims=dims.clone(); // Adjust the data type as needed
    tensor_proto.data_type = DataType::Float as i32;
    tensor_proto.float_data = data.to_vec();
    tensor_proto.name = String::from("data");
    tensor_proto
}