import React, {useState, useContext} from 'react';
import {Form, Button, Accordion, Card, Row, Alert} from 'react-bootstrap';

const SelectInputOutput = ({
                               isInput,
                               onAddNode,
                               nodes,
                               initializers
                           }) => {
    const [node, setNode] = useState('');

    const addNode = () => {
        onAddNode(node);
    };
    return (
        <Accordion defaultActiveKey="0">
            <Accordion.Item eventKey="0">
                <Accordion.Header>Select {isInput ? "Input" : "Output"}</Accordion.Header>
                <Accordion.Body>
                    <Form.Group>
                        <Form.Control
                            as="select"
                            value={node}
                            onChange={(e) => setNode(e.target.value)}
                        >
                            <option value="">Select the {isInput ? "input" : "output"} node</option>
                            {nodes.map((node) => (<option value={node.label}>{node.label}</option>))}
                            {initializers.map((init) => (<option value={init}>{init}</option>))}
                        </Form.Control>
                    </Form.Group>
                    <Button className="mt-3" variant="primary" onClick={addNode}>
                        Add {isInput ? "Input" : "Output"}
                    </Button>
                </Accordion.Body>
            </Accordion.Item>
        </Accordion>
    );
};

export default SelectInputOutput;
