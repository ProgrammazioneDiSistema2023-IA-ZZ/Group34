import React, {useState, useEffect} from 'react';
import FixedGraph from "./Graph";
import NeuralNetwork from "./NeuralNetwork";

//import './App.css';

function App() {
    const [message, setMessage] = useState('');
    const [graph, setGraph] = useState({
        nodes: [
            {id: 1, label: 'Node 1'},
            {id: 2, label: 'Node 2'},
            {id: 3, label: 'Node 3'},
            {id: 4, label: 'Node 4'},
            {id: 5, label: 'Node 5'},
            {id: 6, label: 'Node 6'},
        ],
        edges: [
            {from: 1, to: 2},
            {from: 1, to: 3},
            {from: 2, to: 4},
            {from: 3, to: 5},
            {from: 4, to: 6},
            {from: 5, to: 6},
        ],
    });
    console.log(graph)

    useEffect(() => {
        fetch('http://localhost:3001/get')
            .then(response => response.json())
            .then((data) => {
                setMessage(data.message)
                setGraph(() => JSON.parse(data.graph))

            })
            .catch(error => console.error('Error:', error));
    }, []);


    return (
        <div className="App">
            <header className="App-header">
                <p>Message: {message}</p>

                <div>
                    <h1>Graph Visualization Example</h1>
                    {/*<FixedGraph graph={graph}/>*/}
                    {graph && graph.nodes.length > 10 && <NeuralNetwork graph={graph}/>}

                </div>
            </header>
        </div>
    );
}

export default App;
