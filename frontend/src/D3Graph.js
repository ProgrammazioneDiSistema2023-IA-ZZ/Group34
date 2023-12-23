import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const D3Graph = () => {
    const graphRef = useRef(null);

    useEffect(() => {
        const width = 400;
        const height = 200;

        const nodes = [
            { id: 1, x: 50, y: 50 },
            { id: 2, x: 150, y: 50 },
            { id: 3, x: 250, y: 50 },
            { id: 4, x: 150, y: 150 },
        ];

        const links = [
            { source: 1, target: 4 },
            { source: 2, target: 4 },
            { source: 3, target: 4 },
        ];

        const svg = d3.select(graphRef.current).attr('width', width).attr('height', height);

        const link = svg
            .selectAll('line')
            .data(links)
            .enter()
            .append('line')
            .attr('stroke', '#000')
            .attr('stroke-width', 2);

        const node = svg
            .selectAll('rect')
            .data(nodes)
            .enter()
            .append('rect')
            .attr('width', 80)
            .attr('height', 40)
            .attr('fill', 'lightblue');

        const simulation = d3
            .forceSimulation(nodes)
            .force('link', d3.forceLink(links).id((d) => d.id))
            .force('charge', d3.forceManyBody())
            .force('center', d3.forceCenter(width / 2, height / 2));

        simulation.on('tick', () => {
            link
                .attr('x1', (d) => d.source.x)
                .attr('y1', (d) => d.source.y)
                .attr('x2', (d) => d.target.x)
                .attr('y2', (d) => d.target.y);

            node.attr('x', (d) => d.x - 40).attr('y', (d) => d.y - 20);
        });

        return () => {
            simulation.stop();
        };
    }, []);

    return <svg ref={graphRef}></svg>;
};

export default D3Graph;
