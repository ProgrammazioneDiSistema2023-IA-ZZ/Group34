import React, { useState } from 'react';
import { Modal, Button } from 'react-bootstrap';

const RunResultsModal = ({ show, onHide, result }) => {
    return (
        <Modal show={show} onHide={onHide}>
            <Modal.Header closeButton>
                <Modal.Title>Run Results</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <p>{result}</p>
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
