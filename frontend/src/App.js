import React, {useState, useEffect} from 'react';
import FixedGraph from "./Graph";
import NeuralNetwork from "./NeuralNetwork";
import {Button, Col, Container, Row, Form} from "react-bootstrap";
import 'bootstrap/dist/css/bootstrap.min.css';
import {FaFileDownload, FaPlay} from "react-icons/fa";

//import './App.css';

function App() {
    const [message, setMessage] = useState('');
    const [selectedGraph, setSelectedGraph] = useState(null);
    const [graph, setGraph] = useState({nodes: [], edges: []});
    console.log(graph)

    function onLoad(){
        fetch('http://localhost:3001/model/' + selectedGraph)
            .then(response => response.json())
            .then((data) => {
                setMessage(data.message)
                setGraph(() => JSON.parse(data.graph))

            })
            .catch(error => console.error('Error:', error));
    }
    function onRun(){
        fetch('http://localhost:3001/model/run')
            .then(response => response.json())
            .then((data) => {
                console.log({data})
            })
            .catch(error => console.error('Error:', error));
    }


    let usenn = true; //change this flag to show the alternative layout


    return (
        <div className="App">

            <Container fluid>
                <Row className="mb-4">
                    <Col>
                        <h1 className="text-center">Onnx Runtime Environment</h1>
                    </Col>
                </Row>

                <Row className="mb-4">
                    <Col className="text-center col-lg-10">
                        <Form.Group controlId="selectGraph">
                            <Form.Label>Select Graph:</Form.Label>
                            <Form.Control as="select" onChange={(e) => setSelectedGraph(e.target.value)}
                                          value={selectedGraph}>
                                <option value={null}>-- Select Graph --</option>
                                <option value="1">Mobilenet</option>
                                <option value="2">Resnet</option>
                                <option value="3">Squeezenet</option>
                                <option value="4">Caffenet</option>
                                <option value="5">Alexnet</option>
                            </Form.Control>
                        </Form.Group>
                    </Col>
                    <Col className="text-center col-lg-1">
                        <Button variant="primary" className="mr-2" onClick={onLoad}>
                            <FaFileDownload className="mr-1"/> Load
                        </Button>
                    </Col>
                    <Col className="text-center col-lg-1">
                        <Button variant="success" onClick={onRun}>
                            <FaPlay className="mr-1"/> Play
                        </Button>
                    </Col>
                </Row>

                <Row className="mt-4">
                    <Col>
                        {graph && graph.nodes.length > 10 && (
                            <div className="border p-3">
                                <NeuralNetwork graph={graph}/>
                            </div>
                        )}
                    </Col>
                </Row>
            </Container>
        </div>
    );
}

export default App;
