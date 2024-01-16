import React, {useState, useCallback, useEffect} from "react";
import ReactFlow, {
    addEdge,
    useNodesState,
    useEdgesState,
    ConnectionLineType,
} from "react-flow-renderer";
import {Button, Form, Modal, Spinner} from "react-bootstrap";

function NeuralNetwork({graph}) {
    const verticalSpacing = 10;
    const [selectedNode, setSelectedNode] = useState(null);

    let initialNodes = graph && graph.nodes.map((node) => {
        return {
            data: {
                label: node.label
            },
            id: node.id + "",
            position: {x: 100, y: 20}
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
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if(graph){
            setNodes(graph.nodes.map((node) => {
                return {
                    data: {
                        label: node.label
                    },
                    id: node.id + "",
                    position: {x: 100, y: 20}
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
                        style: {stroke: "red"},
                    },
                    eds
                )
            ),
        [setEdges]
    );


    useEffect(() => {
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

        updateParallelNodesPosition();
    }, [nodes, edges, setNodes]);

    const getNodeId = () => Math.random();

    const onInit = () => {
        setLoading(false);
        console.log("Logged");
    };

    const displayCustomNamedNodeModal = () => {
        setIsModalVisible(true);
    };

    const handleCancel = () => {
        setSelectedNode(null)
        setIsModalVisible(false);
    };

    const handleOk = (data) => {
        onAdd(data.nodeName);
        setIsModalVisible(false);
    };

    const onAdd = (data) => {
        const newNode = {
            id: String(getNodeId()),
            data: {label: data},
            position: {
                x: 100,
                y: 100 + nodes.length * verticalSpacing,
            },
        };
        setNodes((nds) => nds.concat(newNode));
    };

    return (
        <>
            <div style={{height: '80vh', margin: '10px'}}>

                <Button variant="primary" onClick={displayCustomNamedNodeModal}>
                    Add Custom Name Node
                </Button>

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
                    onNodeClick={(event, node) => {
                        setIsModalVisible(true)
                        setSelectedNode(node);
                    }}
                />


            </div>

            <Modal show={isModalVisible} size={"lg"} onHide={handleCancel}>
                <Modal.Header closeButton>
                    <Modal.Title>{selectedNode ? "Edit node: " + selectedNode.data.label : "Create new node"}</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <p>Modal content goes here...</p>
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={handleCancel}>
                        Close
                    </Button>
                    <Button variant="primary" onClick={handleCancel}>
                        Save Changes
                    </Button>
                </Modal.Footer>
            </Modal>
        </>
    );
}

export default NeuralNetwork;
