import React from 'react';

const RectangularNode = ({ nodeData }) => {
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

export default RectangularNode;
