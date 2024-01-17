import React, {useState} from 'react';
import {Modal, Button, Form, Switch, Image, Spinner} from 'react-bootstrap';
import ape from './images/ape.jpg';
import aquila from './images/aquila.jpg';
import gatto from './images/gatto.jpg';

const RunModal = ({show, handleRun, handleClose}) => {
    const [useDefaultInput, setUseDefaultInput] = useState(true);
    const [selectedImage, setSelectedImage] = useState('');
    const [useParallelization, setUseParallelization] = useState(false);
    const [loading, setLoading] = useState(false);

    const customImages = [
        {id: 0, url: ape, label: 'Image 1'},
        {id: 1, url: aquila, label: 'Image 2'},
        {id: 2, url: gatto, label: 'Image 3'},
    ];

    return (
        <Modal show={show} onHide={handleClose} size={"lg"}>
            <Modal.Header closeButton>
                <Modal.Title>Run Network</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                {loading ?
                    <div className="d-flex align-items-center justify-content-center vh-20">
                        <Spinner animation="border" variant="primary" role="status">
                            <span className="sr-only"/>
                        </Spinner>
                    </div>
                    :
                    <Form>
                        <Form.Group controlId="useDefaultInput">
                            <Form.Check
                                type="switch"
                                label="Use Default Input"
                                checked={useDefaultInput}
                                onChange={() => setUseDefaultInput(!useDefaultInput)}
                            />
                        </Form.Group>

                        {!useDefaultInput && (
                            <Form.Group controlId="selectImage">
                                <Form.Label className={"h4"}>Select Image</Form.Label>
                                <div style={{display: 'flex', flexDirection: 'row'}}>
                                    {customImages.map((image) => (
                                        <div key={image.id} style={{marginRight: '20px', textAlign: 'center'}}>
                                            <Image src={image.url} alt={image.label} width={200} height={150}/>
                                            <Form.Check
                                                type="radio"
                                                label={image.label}
                                                checked={selectedImage === image.id}
                                                onChange={() => setSelectedImage(image.id)}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </Form.Group>
                        )}

                        <Form.Group controlId="useParallelization">
                            <Form.Label className={"h4"}>Use Parallelization</Form.Label>
                            <Form.Check
                                type="switch"
                                id="parallelization-switch"
                                label="Use Parallelization"
                                checked={useParallelization}
                                onChange={() => setUseParallelization(!useParallelization)}
                            />
                        </Form.Group>
                    </Form>}
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={handleClose}>
                    Close
                </Button>
                <Button variant="primary" onClick={()=>{
                    setLoading(true);
                    handleRun({use_default: useDefaultInput, image: parseInt(selectedImage) || 0, use_parallelization: useParallelization}, setLoading);
                }}>
                    Run
                </Button>
            </Modal.Footer>
        </Modal>
    );
};

export default RunModal;
