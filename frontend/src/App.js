import React, {useState, useEffect} from 'react';
import FixedGraph from "./Graph";
//import './App.css';

function App() {
    const [message, setMessage] = useState('');
    console.log({message})

    useEffect(() => {
        console.log("CIAO")
        fetch('http://localhost:3001/get')
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
