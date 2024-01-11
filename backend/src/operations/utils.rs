#[allow(unused_imports)]
use crate::onnx::{attribute_proto, tensor_proto};
use crate::utils::OnnxError;
#[allow(unused_imports)]
use crate::onnx::{
        attribute_proto::AttributeType, tensor_proto::DataType, AttributeProto, GraphProto,
        NodeProto, TensorProto,
    };
use ndarray::*;
use std::collections::HashMap;

#[allow(dead_code)]
pub enum TensorValue {
    Float(Vec<f32>),
    UInt8(Vec<u8>),
    Int8(Vec<i8>),
    UInt16(Vec<u16>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    String(Vec<String>),
    Bool(Vec<bool>),
    Double(Vec<f64>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
}

//------------------------------------tensor traits impl----------------------------------------------------
pub trait TensorType {
    type DataType;

    //convert tensor to array
    fn tensor_to_array(tensor: &TensorProto) -> Result<ArrayD<Self::DataType>, OnnxError>;

    //convert array to tensor
    fn array_to_tensor(array: ArrayD<Self::DataType>) -> TensorValue;
}

impl TensorType for f32 {
    type DataType = f32;

    fn tensor_to_array(tensor: &TensorProto) -> Result<ArrayD<Self::DataType>, OnnxError> {
        // Extract shape from the tensor.
        let shape: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();

        if !tensor.float_data.is_empty() {
            ArrayD::from_shape_vec(shape, tensor.float_data.clone())
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else if !tensor.raw_data.is_empty() {
            let data = parse_raw_data_as_floats(&tensor.raw_data);
            ArrayD::from_shape_vec(shape, data).map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for FLOAT type".to_string(),
            ))
        }
    }

    fn array_to_tensor(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::Float(array.into_dyn().into_raw_vec())
    }
}

impl TensorType for i32 {
    type DataType = i32;

    fn tensor_to_array(tensor: &TensorProto) -> Result<ArrayD<Self::DataType>, OnnxError> {
        let shape: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();

        if !tensor.int32_data.is_empty() {
            ArrayD::from_shape_vec(shape, tensor.int32_data.clone())
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for INT32 type".to_string(),
            ))
        }
    }

    fn array_to_tensor(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::Int32(array.into_dyn().into_raw_vec())
    }
}

impl TensorType for i64 {
    type DataType = i64;

    fn tensor_to_array(tensor: &TensorProto) -> Result<ArrayD<Self::DataType>, OnnxError> {
        // Extract shape from the tensor.
        let shape: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();

        if !tensor.int64_data.is_empty() {
            ArrayD::from_shape_vec(shape, tensor.int64_data.clone())
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else if !tensor.raw_data.is_empty() {
            // Parse raw data as floats.
            let data = parse_raw_data_as_ints64(&tensor.raw_data);
            ArrayD::from_shape_vec(shape, data).map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for INT64 type".to_string(),
            ))
        }
    }

    fn array_to_tensor(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::Int64(array.into_dyn().into_raw_vec())
    }
}

impl TensorType for String {
    type DataType = String;

    fn tensor_to_array(tensor: &TensorProto) -> Result<ArrayD<Self::DataType>, OnnxError> {
        // Extract shape from the tensor.
        let shape: Vec<usize> = tensor.dims.iter().map(|&dim| dim as usize).collect();

        // Check if string_data is present and matches the expected length.
        if !tensor.string_data.is_empty() {
            let string_data = tensor
                .string_data
                .iter()
                .map(|s| String::from_utf8_lossy(s).to_string())
                .collect::<Vec<_>>();
            ArrayD::from_shape_vec(shape, string_data)
                .map_err(|e| OnnxError::ShapeMismatch(e.to_string()))
        } else {
            Err(OnnxError::InvalidValue(
                "No valid data found for STRING type".to_string(),
            ))
        }
    }

    fn array_to_tensor(array: ArrayD<Self::DataType>) -> TensorValue {
        TensorValue::String(array.into_dyn().into_raw_vec())
    }
}
//-------------------------------------------------------------------------------------------------------------------------

pub fn ndarray_to_tensor_proto<T: TensorType>(
    result: ArrayD<T::DataType>,
    output_name: &str,
) -> Result<TensorProto, OnnxError> {
    // Extract dimensions from the NDArray and convert them to i64.
    let tensor_dims = result
        .shape()
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<i64>>();

    // Convert NDArray data to tensor data.
    let tensor_data = T::array_to_tensor(result);

    // Construct the TensorProto.
    Ok(make_tensor(
        output_name.to_string(),
        tensor_dims,
        tensor_data,
    ))
}

pub fn convert_to_output_tensor(
    node: &NodeProto,
    result: ArrayD<f32>,
) -> Result<TensorProto, OnnxError> {
    let output_name = node
        .output
        .get(0)
        .ok_or(OnnxError::InternalError("Output name missing".to_string()))?;

    ndarray_to_tensor_proto::<f32>(result, output_name)
}

pub fn tensor_proto_to_ndarray<T: TensorType>(
    tensor: &TensorProto,
) -> Result<ArrayD<T::DataType>, OnnxError> {
    T::tensor_to_array(tensor)
}

///from attribute poroto to hasmap of attribute
pub fn extract_attributes(
    attributes: &[AttributeProto],
) -> Result<HashMap<String, Attribute<String>>, OnnxError> {
    let mut attribute_map = HashMap::new();

    for attr in attributes {
        // Extract the attribute name.
        let key = attr.name.to_string();

        // Match on the attribute type and extract the value accordingly.
        let value = match attr.r#type {
            x if x == attribute_proto::AttributeType::Float as i32 => Attribute::Float(attr.f),
            x if x == attribute_proto::AttributeType::Int as i32 => Attribute::Int(attr.i),
            x if x == attribute_proto::AttributeType::String as i32 => {
                Attribute::String(String::from_utf8(attr.s.to_vec()).map_err(|_| {
                    OnnxError::ConversionError(
                        "Failed to convert bytes to UTF-8 string".to_string(),
                    )
                })?)
            }
            x if x == attribute_proto::AttributeType::Tensor as i32 => {
                Attribute::Tensor(attr.t.clone().unwrap().clone())
            }
            x if x == attribute_proto::AttributeType::Graph as i32 => {
                Attribute::Graph(attr.g.clone().unwrap().clone())
            }
            x if x == attribute_proto::AttributeType::Floats as i32 => {
                Attribute::Floats(attr.floats.clone())
            }
            x if x == attribute_proto::AttributeType::Ints as i32 => {
                Attribute::Ints(attr.ints.clone())
            }
            x if x == attribute_proto::AttributeType::Strings as i32 => Attribute::Strings(
                attr.strings
                    .iter()
                    .map(|s| {
                        String::from_utf8(s.to_vec()).map_err(|_| {
                            OnnxError::ConversionError(
                                "Failed to convert bytes to UTF-8 string".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            x if x == AttributeType::Tensor as i32 => Attribute::Tensors(attr.tensors.to_vec()),
            x if x == AttributeType::Graphs as i32 => Attribute::Graphs(attr.graphs.to_vec()),
            _ => {
                return Err(OnnxError::UnsupportedOperation(
                    "Unsupported attribute type".to_string(),
                ))
            }
        };

        // Insert the extracted attribute into the map.
        attribute_map.insert(key, value);
    }

    Ok(attribute_map)
}

///extract attribute from hasmap
pub fn get_float_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<f32>,
) -> Result<f32, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_float())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

///extract attribute from hasmap
pub fn get_int_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<i64>,
) -> Result<i64, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_int())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}
///extract attribute from hasmap
pub fn get_ints_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<Vec<i64>>,
) -> Result<Vec<i64>, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_ints().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// Retrieves a String attribute.
///
/// # Arguments
///
/// * `attributes` - The HashMap of attributes.
/// * `key` - The name of the attribute to retrieve.
/// * `default_value` - An optional default String to use if the attribute doesn't exist.
///
/// # Returns
///
/// * `Result<String, OnnxError>` - The extracted String attribute or an error.
pub fn get_string_attribute(
    attributes: &HashMap<String, Attribute<String>>,
    key: &str,
    default_value: Option<String>,
) -> Result<String, OnnxError> {
    attributes
        .get(key)
        .and_then(|attr| attr.as_string().cloned())
        .or(default_value)
        .ok_or(OnnxError::AttributeNotFound(key.to_string()))
}

/// padding in 2d matrix
pub fn pad_matrix_2d(matrix: &Array2<f32>, pads: &Vec<i64>) -> Result<Array2<f32>, OnnxError> {
    // Extract padding values
    let top = pads[0] as usize;
    let left = pads[1] as usize;
    let bottom = pads[2] as usize;
    let right = pads[3] as usize;

    // If no padding is needed, return a clone of the input matrix
    if top == 0 && bottom == 0 && left == 0 && right == 0 {
        return Ok(matrix.clone());
    }

    // Calculate the shape of the padded matrix
    let padded_shape = (
        matrix.shape()[0] + top + bottom,
        matrix.shape()[1] + left + right,
    );

    // Create a new matrix filled with zeros for padding
    let mut padded_matrix = Array2::zeros(padded_shape);

    // Slice the padded matrix and assign the input matrix values to it
    padded_matrix
        .slice_mut(s![
            top..padded_shape.0 - bottom,
            left..padded_shape.1 - right
        ])
        .assign(&matrix);

    // Return the padded matrix
    Ok(padded_matrix)
}

pub fn pad_matrix_3d(matrix: &Array3<f32>, pads: &Vec<i64>) -> Result<Array3<f32>, OnnxError> {
    let c = matrix.shape()[0];

    // Vector to store the actual padded slices (owned data)
    let mut padded_data = Vec::with_capacity(c);

    for depth in 0..c {
        // Extract the 2D slice from the 3D array
        let slice = matrix.index_axis(ndarray::Axis(0), depth).to_owned();

        // Pad the slice using the existing function
        let padded = pad_matrix_2d(&slice, pads)?;

        // Push the padded slice into the vector
        padded_data.push(padded);
    }

    // Derive a vector of views from our owned data
    let padded_slices: Vec<_> = padded_data.iter().map(|array| array.view()).collect();

    // Now, directly use the stack function with the slice of views
    ndarray::stack(ndarray::Axis(0), &padded_slices[..])
        .map_err(|_| OnnxError::InternalError("Failed to stack matrices.".to_string()))
}

pub fn stack_along_batch_dimension<T, D>(
    tensors: Vec<ArrayBase<OwnedRepr<T>, D>>,
) -> Result<ArrayD<T>, OnnxError>
where
    T: Clone,
    D: Dimension,
{
    // Create views of the tensors
    let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();

    // Stack along the batch dimension
    let stacked_output = ndarray::stack(Axis(0), &views).map_err(|_| {
        OnnxError::ShapeError("Failed to stack tensors along batch dimension.".to_string())
    })?;

    // Construct the result dimensions dynamically
    let mut result_dims: Vec<usize> = Vec::with_capacity(stacked_output.ndim());
    result_dims.push(tensors.len()); // First dimension is the batch size
    result_dims.extend_from_slice(stacked_output.shape().split_at(1).1); // Skip the first dimension as we already added it

    let reshaped_output =
        ArrayD::from_shape_vec(IxDyn(&result_dims), stacked_output.into_iter().collect())
            .map_err(|_| OnnxError::ShapeError("Failed to reshape stacked output.".to_string()))?;

    Ok(reshaped_output)
}

pub fn parse_raw_data_as_floats(raw_data: &[u8]) -> Vec<f32> {
    let mut doubles = Vec::with_capacity(raw_data.len() / 4);

    for i in (0..raw_data.len()).step_by(4) {
        let bytes = [
            raw_data[i],
            raw_data[i + 1],
            raw_data[i + 2],
            raw_data[i + 3],
        ];
        let double_value = f32::from_le_bytes(bytes);
        doubles.push(double_value);
    }

    doubles
}

pub fn parse_raw_data_as_ints64(raw_data: &[u8]) -> Vec<i64> {
    let mut ints64 = Vec::with_capacity(raw_data.len() / 8);

    for i in (0..raw_data.len()).step_by(8) {
        let bytes = [
            raw_data[i],
            raw_data[i + 1],
            raw_data[i + 2],
            raw_data[i + 3],
            raw_data[i + 4],
            raw_data[i + 5],
            raw_data[i + 6],
            raw_data[i + 7],
        ];
        let int64_value = i64::from_le_bytes(bytes);
        ints64.push(int64_value);
    }

    ints64
}

impl TensorProto {
    pub fn new() -> TensorProto {
        ::std::default::Default::default()
    }
}

pub fn make_tensor(name: String, dims: Vec<i64>, vals: TensorValue) -> TensorProto {
    let mut tensor_proto = TensorProto::new();
    tensor_proto.dims = dims;
    tensor_proto.name = name;

    match vals {
        TensorValue::Float(vals) => {
            tensor_proto.float_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Float as i32;
        }
        TensorValue::UInt8(vals) => {
            tensor_proto.int32_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Uint8 as i32;
        }
        TensorValue::Int8(vals) => {
            tensor_proto.int32_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Int8 as i32;
        }
        TensorValue::UInt16(vals) => {
            tensor_proto.int32_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Uint16 as i32;
        }
        TensorValue::Int16(vals) => {
            tensor_proto.int32_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Int16 as i32;
        }
        TensorValue::Int32(vals) => {
            tensor_proto.int32_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Int32 as i32;
        }
        TensorValue::String(vals) => {
            tensor_proto.string_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::String as i32;
        }
        TensorValue::UInt32(vals) => {
            tensor_proto.uint64_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Uint32 as i32;
        }
        TensorValue::UInt64(vals) => {
            tensor_proto.uint64_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Uint64 as i32;
        }
        TensorValue::Int64(vals) => {
            tensor_proto.int64_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Int64 as i32;
        }
        TensorValue::Double(vals) => {
            tensor_proto.double_data = vals.into_iter().map(Into::into).collect();
            tensor_proto.data_type = DataType::Double as i32;
        }
        //true, false = 1,0
        TensorValue::Bool(vals) => {
            tensor_proto.int32_data = vals.into_iter().map(|v| if v { 1 } else { 0 }).collect();
            tensor_proto.data_type = DataType::Bool as i32;
        }
    }
    tensor_proto
}

#[derive(Debug)]
pub enum Attribute<S> {
    Float(f32),
    Floats(Vec<f32>),
    Int(i64),
    Ints(Vec<i64>),
    String(S),
    Strings(Vec<S>),
    Tensor(TensorProto),
    Tensors(Vec<TensorProto>),
    Graph(GraphProto),
    Graphs(Vec<GraphProto>),
}

#[allow(dead_code)]
impl<S: std::fmt::Display> Attribute<S> {
    pub fn as_float(&self) -> Option<f32> {
        if let Attribute::Float(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    pub fn as_floats(&self) -> Option<&Vec<f32>> {
        if let Attribute::Floats(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        if let Attribute::Int(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    pub fn as_ints(&self) -> Option<&Vec<i64>> {
        if let Attribute::Ints(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_string(&self) -> Option<&S> {
        if let Attribute::String(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_strings(&self) -> Option<&Vec<S>> {
        if let Attribute::Strings(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_tensor(&self) -> Option<&TensorProto> {
        if let Attribute::Tensor(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_tensors(&self) -> Option<&Vec<TensorProto>> {
        if let Attribute::Tensors(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_graph(&self) -> Option<&GraphProto> {
        if let Attribute::Graph(ref value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_graphs(&self) -> Option<&Vec<GraphProto>> {
        if let Attribute::Graphs(ref value) = self {
            Some(value)
        } else {
            None
        }
    }
}
