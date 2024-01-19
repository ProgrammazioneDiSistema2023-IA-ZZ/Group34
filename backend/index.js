const express = require('express');
const cors = require('cors');
const app = express();
const port = 3001;

//get rust functions
const rust = require('.');
console.log(rust.hello());
console.log(rust.start());

app.use(cors());
app.use(express.json());


app.get('/model/:ordinal', (req, res) => {
  res.json({graph: rust.select_model(parseInt(req.params.ordinal))});
});

app.post('/runmodel', (req, res) => {
  res.json({result: rust.run(JSON.stringify(req.body))});
});
app.get('/node/:nodeName', (req, res) => {
  res.json({graph: rust.get_node_js(req.params.nodeName)});
});

app.post('/node', (req, res) => {
  res.json({graph: rust.create_node(JSON.stringify(req.body))});
});

app.put('/node', (req, res) => {
  res.json({graph: rust.modify_node(JSON.stringify(req.body))});
});

app.delete('/node/:nodeName', (req, res) => {
  res.json({graph: rust.delete_node(req.params.nodeName)});
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

