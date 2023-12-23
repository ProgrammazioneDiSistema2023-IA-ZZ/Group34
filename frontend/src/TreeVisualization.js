import React from 'react';
import Tree from 'react-d3-tree';

const TreeNode = ({ nodeData }) => {
    const width = 120;
    const height = 40;

    const rectStyle = {
        fill: nodeData.color || 'lightblue',
        stroke: 'black',
        strokeWidth: 2,
    };

    return (
        <g transform={`translate(${-width / 2},${-height / 2})`}>
            <rect width={width} height={height} style={rectStyle} />
            <text x={width / 2} y={height / 2} dominantBaseline="middle" textAnchor="middle" fill="white">
                {nodeData.name}
            </text>
        </g>
    );
};

const TreeVisualization = () => {
    const treeData = {
        name: 'Root',
        attributes: {
            color: 'gray',
        },
        children: [
            {
                name: 'Combined Node',
                attributes: {
                    color: 'purple',
                },
                children: [
                    { name: 'Node 1', color: 'red' },
                    { name: 'Node 2', color: 'blue' },
                ],
            },
            {
                name: 'Node 3',
                color: 'green',
                children: [
                    { name: 'Node 1', color: 'red' },
                    { name: 'Node 2', color: 'blue' },
                ],
            },
            {
                name: 'Node 4',
                color: 'orange',
            },
        ],
    };

    const treeConfig = {
        nodeSvgShape: {
            shape: 'none',
        },
        nodeSize: {
            x: 150,
            y: 60,
        },
        renderCustomNodeElement: ({ nodeDatum, toggleNode }) => (
            <g onClick={() => toggleNode(nodeDatum)}>
                <TreeNode nodeData={nodeDatum} />
            </g>
        ),
    };

    return (
        <div style={{ width: '100%', height: '500px' }}>
            <Tree data={treeData} orientation="vertical" translate={{ x: 200, y: 100 }} separation={{ siblings: 1.2, nonSiblings: 1.2 }} {...treeConfig} />
        </div>
    );
};

export default TreeVisualization;
