import React, { useState, useCallback, useEffect } from "react";
import ReactFlow, {
    addEdge,
    useNodesState,
    useEdgesState,
    ConnectionLineType,
} from "react-flow-renderer";
import { Button, Form, Modal, Spinner } from "react-bootstrap";
import ModifyNodeModal from "./ModifyNodeModal";

function NeuralNetwork({ graph, setGraph }) {
    const verticalSpacing = 10;
    const [selectedNode, setSelectedNode] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    let initialNodes = graph && graph.nodes.map((node) => {
        return {
            data: {
                label: node.label
            },
            id: node.id + "",
            position: { x: 100, y: 20 }
        }
    });

    let initialEdges = graph && graph.edges.map((edge) => {
        return {
            id: 'e' + edge.from + "-" + edge.to,
            source: edge.from + "",
            target: edge.to + "",
            type: 'smoothstep',
            animated: false
        }
    });

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [Snodes, setSNodes, onSNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [isModify, setIsModify] = useState(false);

    useEffect(() => {
        if (graph) {
            setIsLoading(true);
            setNodes(graph.nodes.map((node) => {
                return {
                    data: {
                        label: node.label
                    },
                    id: node.id + "",
                    position: { x: 100, y: 20 }
                }
            }))
            setEdges(graph.edges.map((edge) => {
                return {
                    id: 'e' + edge.from + "-" + edge.to,
                    source: edge.from + "",
                    target: edge.to + "",
                    type: 'smoothstep',
                    animated: false
                }
            }));
        }
    }, [graph]);

    const onConnect = useCallback(
        (params) =>
            setEdges((eds) =>
                addEdge(
                    {
                        ...params,
                        type: ConnectionLineType.SmoothStep,
                        animated: true,
                        style: { stroke: "red" },
                    },
                    eds
                )
            ),
        [setEdges]
    );

    function onNodeClick(event, node) {
        fetch('http://localhost:3001/node/' + node.data.label)
            .then(response => response.json())
            .then((data) => {
                setIsModify(true)
                setIsModalVisible(true)
                setSelectedNode(JSON.parse(data.graph));
            })
            .catch(error => console.error('Error:', error));
    }


    useEffect(() => {
        updateParallelNodesPosition();
        setIsLoading(false);
    }, [nodes, edges]);

    // Funzione per ottenere gli antenati di un nodo
    const getAncestors = (nodeId) => {
        const ancestors = [];
        const findAncestors = (currentNodeId) => {
            edges.forEach((edge) => {
                if (edge.target === currentNodeId) {
                    ancestors.push(edge.source);
                    findAncestors(edge.source);
                }
            });
        };
        findAncestors(nodeId);
        return ancestors;
    };

    // Funzione per controllare se due nodi sono paralleli
    const areNodesParallel = (node1, node2) => {
        const ancestors1 = getAncestors(node1.id);
        const ancestors2 = getAncestors(node2.id);
        return (
            ancestors1.length === ancestors2.length &&
            ancestors1.every((ancestor, index) => ancestor === ancestors2[index])
        );
    };

    // Funzione per aggiornare la posizione dei nodi paralleli
    const updateParallelNodesPosition = () => {
        const updatedNodes = nodes.map((node) => {
            const parallelNodes = nodes.filter(
                (otherNode) => node.id !== otherNode.id && areNodesParallel(node, otherNode)
            );

            // Calcola la nuova coordinata y per i nodi paralleli
            const newY = parallelNodes.reduce(
                (acc, parallelNode) => Math.max(acc, parallelNode.position.y),
                node.position.y
            );

            //Calcola il numero di figli dell'ancestor
            let ancestor = getAncestors(node.id)[0];
            let n_ancestor_children = edges.filter((edge) => edge.source === ancestor).length;
            if (n_ancestor_children === 0) n_ancestor_children = 1;

            return {
                ...node,
                position: {
                    ...node.position,
                    x: node.position.x + 200 * n_ancestor_children,//todo compute the children of its anchestor
                    y: newY + verticalSpacing + 100 * node.id,
                },
            };
        });

        setSNodes(updatedNodes);
    };

    const onSave = (node) => {
        console.log({ node })
        if (!isModify) {
            fetch('http://localhost:3001/node', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(node),
            })
                .then(response => response.json())
                .then((data) => {
                    console.log(JSON.parse(data.graph))
                    handleCancel();
                    setGraph(() => JSON.parse(data.graph))
                })
                .catch((error) => {
                    console.error('Error:', error);
                });

        } else {
            fetch('http://localhost:3001/node', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(node),
            })
                .then(response => response.json())
                .then((data) => {
                    console.log(JSON.parse(data.graph))
                    handleCancel();
                    setGraph(() => JSON.parse(data.graph))
                })
                .catch((error) => {
                    console.error('Error:', error);
                });

        }
    }

    const getNodeId = () => Math.random();

    const onInit = () => {
        console.log("Logged");
    };

    const displayCustomNamedNodeModal = () => {
        setIsModify(false);
        setIsModalVisible(true);
    };

    const handleCancel = () => {
        setSelectedNode(null)
        setIsModalVisible(false);
    };

    const onAdd = (data) => {
        const newNode = {
            id: String(getNodeId()),
            data: { label: data },
            position: {
                x: 100,
                y: 100 + nodes.length * verticalSpacing,
            },
        };
        setNodes((nds) => nds.concat(newNode));
    };
    const onDelete = (node_name) => {
        fetch('http://localhost:3001/node/' + node_name, {
            method: 'DELETE',
        })
            .then(response => response.json())
            .then((data) => {
                setGraph(() => JSON.parse(data.graph))
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    }

    return (
        <>
            <div style={{ height: '80vh', margin: '10px' }}>

                <Button variant="primary" onClick={displayCustomNamedNodeModal}>
                    Add a node
                </Button>

                {
                    isLoading ?
                        <div className="d-flex align-items-center justify-content-center vh-100">
                            <Spinner animation="border" variant="primary" role="status">
                                <span className="sr-only" />
                            </Spinner>
                        </div>
                        :
                        <ReactFlow
                            nodes={Snodes}
                            edges={edges}
                            onNodesChange={onSNodesChange}
                            onEdgesChange={onEdgesChange}
                            onConnect={onConnect}
                            onInit={onInit}
                            fitView
                            attributionPosition="bottom-left"
                            connectionLineType={ConnectionLineType.SmoothStep}
                            elementsSelectable={true}
                            onNodeClick={onNodeClick}
                        />

                }
            </div>

            <ModifyNodeModal show={isModalVisible} onHide={handleCancel} nodeData={selectedNode} nodes={graph.nodes}
                initializers={graph.initializers} onSave={onSave} isModify={isModify} onDelete={onDelete} />
        </>
    );
}

export default NeuralNetwork;
