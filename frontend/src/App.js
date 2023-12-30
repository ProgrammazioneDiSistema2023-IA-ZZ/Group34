import React, {useState, useEffect} from 'react';
import TreeVisualization from "./TreeVisualization";
import DAGVisualization from "./Graph";
import FixedGraph from "./Graph";
import D3Graph from "./D3Graph";
//import './App.css';

function App() {
    const [message, setMessage] = useState('');

    useEffect(() => {
        fetch('/api/get')
            .then(response => response.json())
            .then(data => setMessage(data.message))
            .catch(error => console.error('Error:', error));
    }, []);


    return (
        <div className="App">
            <header className="App-header">
                <p>Message: {message}</p>

                <div>
                    <h1>Graph Visualization Example</h1>
                    <FixedGraph/>
                </div>
            </header>
        </div>
    );
}

export default App;
