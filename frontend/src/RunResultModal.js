import React, { useState } from 'react';
import {Modal, Button, Row} from 'react-bootstrap';

const RunResultsModal = ({ show, onHide, result }) => {
    return (
        <Modal show={show} onHide={onHide}>
            <Modal.Header closeButton>
                <Modal.Title>Run Results</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                {result && (!result.error ? <>
                    <Row>
                        <h4>Expected:</h4>
                    </Row>
                    <Row>
                        {result.expected.map((ex) => <>
                            <h6>{ex}</h6>
                        </>)}
                    </Row>
                    <Row>
                        <h4>Actual:</h4>
                    </Row>
                    <Row>
                        {result.predicted.map((pr) => <>
                            <h6>{pr}</h6>
                        </>)}</Row>
                    <Row>
                        <h4>Execution time: {parseFloat(result.time).toFixed(3)} seconds</h4>
                    </Row>
                </>:
                <>
                    <h4>Error</h4>
                    <h6>Error running the network</h6>
                </>
                )}
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={onHide}>
                    Close
                </Button>
            </Modal.Footer>
        </Modal>
    );
};

export default RunResultsModal;
