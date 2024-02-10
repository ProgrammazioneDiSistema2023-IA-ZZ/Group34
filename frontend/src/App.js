import React, {useState, useEffect} from 'react';
import FixedGraph from "./Graph";
import NeuralNetwork from "./NeuralNetwork";
import {Button, Col, Container, Row, Form, Spinner} from "react-bootstrap";
import 'bootstrap/dist/css/bootstrap.min.css';
import {FaEye, FaEyeSlash, FaFileDownload, FaPlay} from "react-icons/fa";
import RunModal from "./RunModal";
import RunResultModal from "./RunResultModal";

//import './App.css';

function App() {
    const [message, setMessage] = useState('');
    const [selectedGraph, setSelectedGraph] = useState("");
    const [graph, setGraph] = useState({nodes: [], edges: []});
    const [loading, setLoading] = useState(false);

    function onLoad() {
        setLoading(true);
        fetch('http://localhost:3001/model/' + selectedGraph)
            .then(response => response.json())
            .then((data) => {
                setLoading(false);
                setMessage(data.message)
                setGraph(() => JSON.parse(data.graph))

            })
            .catch(error => console.error('Error:', error));
    }

    function onRun(options, setLoading) {
        console.log({ options });
        fetch('http://localhost:3001/runmodel', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({options}),
        })
            .then(response => response.json())
            .then((data) => {
                console.log({ data });
                setShowRunModal(false);
                setShowRunModalResults(true);
                let results = JSON.parse(data.result);
                results.expected = results.expected.split("|").filter((el) => el !== " " && el !== "");
                results.predicted = results.predicted.split("|").filter((el) => el !== " " && el !== "");
                results.error = false;
                setRunResult(results);
                setLoading(false);
            })
            .catch(error => {
                console.error('Errore:', error);
                setLoading(false);
                setRunResult({expected: [], predicted: [], time: 0, error: true});
            });
    }



    const [showRunModal, setShowRunModal] = useState(false);
    const [showRunModalResults, setShowRunModalResults] = useState(false);
    const [runResult, setRunResult] = useState("");
    const [showGraph, setShowGraph] = useState(false);

    return (
        <div className="App">

            <Container fluid>
                <Row className="mb-4">
                    <Col>
                        <h1 className="text-center">Onnx Runtime Environment</h1>
                    </Col>
                </Row>

                <Row className="mb-4">
                    <Col className="text-center col-md-9">
                        <Form.Group controlId="selectGraph">
                            <Form.Label>Select Graph:</Form.Label>
                            <Form.Control as="select" onChange={(e) => setSelectedGraph(e.target.value)}
                                          value={selectedGraph}>
                                <option value={""}>-- Select Graph --</option>
                                <option value="1">Mobilenet</option>
                                <option value="2">Resnet</option>
                                <option value="3">Squeezenet</option>
                                <option value="4">Caffenet</option>
                                <option value="5">Alexnet</option>
                            </Form.Control>
                        </Form.Group>
                    </Col>
                    <Col className="text-center col-md-1">
                        <Button variant="secondary" className="mr-2 w-100" onClick={() => setShowGraph((old) => !old)}>
                            {showGraph ?
                                <><FaEye className="mr-1"/><br/>  Hide </>
                                :
                                <><FaEyeSlash className="mr-1"/>  <br/>Show </>
                            }
                        </Button>
                    </Col>
                    <Col className="text-center col-md-1">
                        <Button variant="primary" className="mr-2 w-100" onClick={onLoad}>
                            <><FaFileDownload className="mr-1"/> <br/>Load</>
                        </Button>
                    </Col>
                    <Col className="text-center col-md-1">
                        <Button variant="success" className="mr-2 w-100" onClick={() => setShowRunModal(true)}>
                            <FaPlay className="mr-1"/> <br/> Run
                        </Button>
                    </Col>
                </Row>


                {loading && <div className="d-flex align-items-center justify-content-center vh-100">
                    <Spinner animation="border" variant="primary" role="status">
                        <span className="sr-only"/>
                    </Spinner>
                </div>


                }
                {showGraph && <Row className="mt-4">
                    <Col>
                        <div className="border p-3">
                            <NeuralNetwork graph={graph} setGraph={setGraph} setLoading={setLoading}/>
                        </div>
                    </Col>
                </Row>}
                <RunModal show={showRunModal} handleClose={() => {
                    setShowRunModal(false)
                }} handleRun={onRun}/>
                <RunResultModal show={showRunModalResults} onHide={() => setShowRunModalResults(false)}
                                result={runResult}/>

            </Container>
        </div>
    );
}

export default App;
