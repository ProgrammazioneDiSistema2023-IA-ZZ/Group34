import React, {useEffect, useState} from 'react';
import {Modal, Button, Form, Col, Row, Card, ListGroup} from 'react-bootstrap';
import SelectInputOutput from "./SelectInputOutput";
import professors from "dagre/lib/lodash";

const ModifyNodeModal = ({nodeData, show, onHide, onSave, nodes}) => {
    const getEmptyNode = () => {
        return {
            node_name: '',
            input: [],
            output: [],
            operation_type: '',
            domain: '',
            attributes: [],
            doc_string: '',
            name_node_before: '',
            name_node_after: '',
        };
    };
    const [modifiedNode, setModifiedNode] = useState(getEmptyNode());

    useEffect(() => {
        if (nodeData) {
            setModifiedNode({...nodeData});
        }
    }, [nodeData]);

    console.log({modifiedNode})


    const handleInputChange = (attributeIndex, field, value) => {
        const updatedAttributes = [...modifiedNode.attributes];
        updatedAttributes[attributeIndex][field] = value;
        console.log({updatedAttributes})
        setModifiedNode({...modifiedNode, attributes: updatedAttributes});
    };

    const handleAddInput = (selectedInput) => {
        const updatedInputs = [...modifiedNode.input, selectedInput];
        setModifiedNode({...modifiedNode, input: updatedInputs});
    };

    const handleRemoveInput = (index) => {
        const updatedInputs = [...modifiedNode.input];
        updatedInputs.splice(index, 1);
        setModifiedNode({...modifiedNode, input: updatedInputs});
    };

    const handleSave = () => {
        onSave(modifiedNode);
        setModifiedNode(getEmptyNode());
    };

    const handleAddAttribute = () => {
        setModifiedNode((prevNode) => ({
            ...prevNode,
            attributes: [...prevNode.attributes, ["", "", ""]],
        }));
    };

    const handleDeleteAttribute = (index) => {
        setModifiedNode((prevNode) => {
            const newAttributes = [...prevNode.attributes];
            newAttributes.splice(index, 1);
            return {
                ...prevNode,
                attributes: newAttributes,
            };
        });
    };

    return (
        <Modal show={show} onHide={onHide} size="lg">
            <Modal.Header closeButton>
                <Modal.Title>
                    {nodeData ? `Edit Node: ${nodeData.node_name}` : 'Create New Node'}
                </Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Form>
                    <Form.Group>
                        <Form.Label>Node Name</Form.Label>
                        <Form.Control
                            type="text"
                            placeholder="Enter Node Name"
                            value={modifiedNode.node_name}
                            onChange={(e) =>
                                setModifiedNode({...modifiedNode, node_name: e.target.value})
                            }
                        />
                    </Form.Group>

                    <Form.Group>
                        <Form.Label>Operation Type</Form.Label>
                        <Form.Control
                            as="select"
                            value={modifiedNode.operation_type}
                            onChange={(e) =>
                                setModifiedNode({
                                    ...modifiedNode,
                                    operation_type: e.target.value,
                                })
                            }
                        >
                            <option value="">Select Operation Type</option>
                            <option value="Add">Add</option>
                            <option value="Relu">Relu</option>
                            <option value="Exp">Exp</option>
                            <option value="Concat">Concat</option>
                            <option value="Flatten">Flatten</option>
                            <option value="Reshape">Reshape</option>
                            <option value="Conv">Conv</option>
                            <option value="MaxPool">MaxPool</option>
                            <option value="AveragePool">AveragePool</option>
                            <option value="BatchNormalization">BatchNormalization</option>
                            <option value="Dropout">Dropout</option>
                            <option value="Softmax">Softmax</option>
                            <option value="Gemm">Gemm</option>
                            <option value="MatMul">MatMul</option>
                            <option value="ReduceSum">ReduceSum</option>
                            <option value="GlobalAveragePool">GlobalAveragePool</option>
                            <option value="Lrn">Lrn</option>


                        </Form.Control>
                    </Form.Group>

                    <Form.Group>
                        <Form.Label>Domain</Form.Label>
                        <Form.Control
                            type="text"
                            placeholder="Enter Domain"
                            value={modifiedNode.domain}
                            onChange={(e) =>
                                setModifiedNode({...modifiedNode, domain: e.target.value})
                            }
                        />
                    </Form.Group>

                    <Form.Label className="h3 mt-3">Inputs</Form.Label>
                    <Card className={"mb-3"}>
                        <Card.Body>
                            <SelectInputOutput isInput={true} onAddNode={handleAddInput} nodes={nodes}/>
                            <br/>
                            <ListGroup className={"mt-3"}>
                                {modifiedNode.input.map((input, index) => (<ListGroup.Item key={index}>
                                    {input} &nbsp;
                                    <Button
                                        variant="danger"
                                        size="sm"
                                        className="float-right"
                                        onClick={() => handleRemoveInput(index)}
                                    >
                                        Remove
                                    </Button>
                                </ListGroup.Item>))}
                            </ListGroup>
                        </Card.Body>
                    </Card>
                    <Form.Group>
                        <Form.Label className={"h3 mt-3"}>Attributes</Form.Label>
                        <ListGroup>
                            {modifiedNode &&
                                modifiedNode.attributes.length ? <Row>
                                <Col>Name</Col>
                                <Col>Type</Col>
                                <Col>Value</Col>
                                <Col></Col>
                            </Row> : ""}
                            {modifiedNode &&
                                modifiedNode.attributes.map((attribute, index) => (
                                    <Row className={"mb-3"} key={index}>
                                        <Col>
                                            <Form.Control
                                                type="text"
                                                placeholder="Attribute Name"
                                                value={attribute[0]}
                                                onChange={(e) => handleInputChange(index, 0, e.target.value)}
                                            />
                                        </Col>
                                        <Col>
                                            <Form.Control
                                                as="select"
                                                value={attribute[1] + ""}
                                                onChange={(e) => handleInputChange(index, 1, e.target.value)}
                                            >
                                                <option value="">-- Select --</option>
                                                <option value="1">Int</option>
                                                <option value="2">Float</option>
                                                <option value="3">String</option>
                                            </Form.Control>
                                        </Col>
                                        <Col>
                                            <Form.Control
                                                type="text"
                                                placeholder="Attribute Value"
                                                value={attribute[2]}
                                                onChange={(e) => handleInputChange(index, 2, e.target.value)}
                                            />
                                        </Col>
                                        <Col>
                                            <Button variant="danger" onClick={() => handleDeleteAttribute(index)}>
                                                Delete
                                            </Button>
                                        </Col>
                                    </Row>
                                ))}
                        </ListGroup>

                        <Button variant="primary" className="mt-3" onClick={handleAddAttribute}>
                            Add Attribute
                        </Button>
                    </Form.Group>
                </Form>
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={onHide}>
                    Close
                </Button>
                <Button variant="primary" onClick={handleSave}>
                    Save Changes
                </Button>
            </Modal.Footer>
        </Modal>
    );
};

export default ModifyNodeModal;