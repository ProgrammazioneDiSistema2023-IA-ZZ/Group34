import React from 'react';
import Graph from 'react-graph-vis';

const FixedGraph = (graph) => {

    const options = {
        //autoResize: true,
        layout: {
            //improvedLayout: true,
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
            var {nodes, edges} = event;
            console.log('Selected nodes:', nodes);
            console.log('Selected edges:', edges);
        },
    };
    //console.log({graph})

    return (
        <div style={{height: '500px'}}>
            <Graph key={crypto.randomUUID()} graph={graph.graph} options={options} events={events}/>
        </div>
    );
};

export default FixedGraph;
