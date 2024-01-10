import React from 'react';
import Graph from 'react-graph-vis';

const FixedGraph = () => {
    const graph = {
        nodes: [
            { id: 1, label: 'Node 1'},
            { id: 2, label: 'Node 2'},
            { id: 3, label: 'Node 3'},
            { id: 4, label: 'Node 4'},
            { id: 5, label: 'Node 5'},
            { id: 6, label: 'Node 6'},
        ],
        edges: [
            { from: 1, to: 2 },
            { from: 1, to: 3 },
            { from: 2, to: 4 },
            { from: 3, to: 5 },
            { from: 4, to: 6 },
            { from: 5, to: 6 },
        ],
    };

    const options = {
        layout: {
            hierarchical: false
        },
        edges: {
            color: '#000000',
        },
        physics: {
            enabled: false, // Disable physics to prevent node movement
        },
    };

    const events = {
        select: function (event) {
            var { nodes, edges } = event;
            console.log('Selected nodes:', nodes);
            console.log('Selected edges:', edges);
        },
    };
    //console.log({graph})

    return (
        <div style={{ height: '500px' }}>
            <Graph key={crypto.randomUUID()} graph={graph} options={options} events={events} />
        </div>
    );
};

export default FixedGraph;
