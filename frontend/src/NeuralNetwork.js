import React, { useState, useCallback, useEffect } from "react";
import ReactFlow, {
    addEdge,
    useNodesState,
    useEdgesState,
    ConnectionLineType,
} from "react-flow-renderer";
import { Button, Modal, Input, Form } from "antd";

function NeuralNetwork({graph}) {
    const numNodes = 35;
    const verticalSpacing = 10;

    /*
    const initialNodes = Array.from({ length: numNodes }, (_, index) => ({
        id: `${index + 1}`,
        data: {
            label: `Node ${index + 1}`,
        },
        position: { x: 100, y: 100 + index * verticalSpacing },
    }));
    */

    /*
    const initialEdges = initialNodes.map((node, index) => ({
        id: `e1-${index + 2}`,
        source: "1",
        target: `${index + 2}`,
        type: "smoothstep",
        animated: true,
    }));
     */

    const initialNodes = graph && graph.nodes.map((node)=>{
        return {
            data: {
                label: node.label
            },
            id: node.id + "",
            position: {x:100, y:100}
        }
    });

    const initialEdges = graph && graph.edges.map((edge) => {
        return {
            id: 'e'+edge.from+"-"+edge.to,
            source: edge.from + "",
            target: edge.to + "",
            type: 'smoothstep',
            animated: false}
    });

    console.log({initialNodes})
    console.log({initialEdges})

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [Snodes, setSNodes, onSNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [isModalVisible, setIsModalVisible] = useState(false);

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

                return {
                    ...node,
                    position: {
                        ...node.position ,
                        x: node.position.x + 50 * getAncestors(node.id) && getAncestors(node.id)[0].id,//todo compute the children of its anchestor
                        y: newY + verticalSpacing + 50 * node.id,
                    },
                };
            });

            setSNodes(updatedNodes);
        };

        updateParallelNodesPosition();
    }, [nodes, edges, setNodes]);

    const getNodeId = () => Math.random();

    const onInit = () => {
        console.log("Logged");
    };

    const displayCustomNamedNodeModal = () => {
        setIsModalVisible(true);
    };

    const handleCancel = () => {
        setIsModalVisible(false);
    };

    const handleOk = (data) => {
        onAdd(data.nodeName);
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

    return (
        <div style={{ height: "100vh", margin: "10px" }}>
            <Modal
                title="Basic Modal"
                open={isModalVisible}
                onCancel={handleCancel}
            >
                <Form onFinish={handleOk} autoComplete="off" name="new node">
                    <Form.Item label="Node Name" name="nodeName">
                        <Input />
                    </Form.Item>

                    <Form.Item>
                        <Button type="primary" htmlType="submit">
                            Submit
                        </Button>
                    </Form.Item>
                </Form>
            </Modal>
            <Button type="primary" onClick={displayCustomNamedNodeModal}>
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
            />
        </div>
    );
}

export default NeuralNetwork;
