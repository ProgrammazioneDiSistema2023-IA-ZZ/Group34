mod onnx {
    include!("onnx.rs");
}

use std::io::ErrorKind;
use ndarray::{arr2, Array, Array2, Array4, ArrayD};
use protoc::Error;
use tract_onnx::pb::AttributeProto;
use tract_onnx::prelude::tract_itertools::Itertools;
use onnx::ModelProto;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::TensorProto;

fn main() {
    // Load and parse your ProtoBuf file (e.g., "squeezenet.onnx")
    let data = std::fs::read("src/squeezenet.onnx").expect("Failed to read ProtoBuf file");
    let parsed_proto: ModelProto = prost::Message::decode(&data[..]).expect("Failed to decode ProtoBuf data");

    // Use the parsed ProtoBuf data as needed
    let i = parsed_proto.graph.unwrap().initializer.get(10).unwrap().clone();
    let j = i.clone();
    let matrix = Array4::from_shape_vec((64, 16,1,1), i.float_data).unwrap();
    println!("{:?}", op_add(vec![i, j], vec![]));


}

struct Operation{
    op_type: OperationType,
    input: Vec<TensorProto>,
    op_attributes: Vec<AttributeProto>
}

enum OperationType{
    ADD,
    RELU
}

fn perform_operation(op: Operation) -> Result<TensorProto, Error>{
    match op.op_type {
        OperationType::ADD => {
            op_add(op.input, op.op_attributes)
        }
        _ => {
            Err(Error::new(ErrorKind::Other, "Incorrect operation!"))
        }
    }
}

fn op_add(input: Vec<TensorProto>, op_attributes: Vec<AttributeProto>) -> Result<TensorProto, Error>{
    let a: ArrayD<f32> = input.get(0).unwrap().into();
    let b: ArrayD<f32> = input.get(0).unwrap().into();
    let c = a+b;
    println!("{:?}", c);
    return Ok(TensorProto::from(c));
    
}

impl <T> Into<ArrayD<T>> for TensorProto {
    fn into(self) -> ArrayD<T> {
        let shape = self.dims;
        match self.data_type {
            1 => {
                ArrayD::from_shape_vec(&shape.into_iter().collect_tuple().unwrap(), self.float_data).unwrap()
            }
            _ => {
                panic!("Type not defined")
            }
        }
    }
}