const express = require('express');
const cors = require('cors');
const app = express();
const port = 3001;

//get rust functions
const rust = require('.');
console.log(rust.hello());
console.log(rust.start());

app.use(cors());


app.get('/model/:ordinal', (req, res) => {
  console.log(req.params.ordinal);
  res.json({graph: rust.select_model(parseInt(req.params.ordinal))});
});

app.get('/model/run', (req, res) => {
  res.json({result: rust.run()});
});
app.get('/node/:nodeName', (req, res) => {
  res.json({graph: rust.get_node_js(req.params.nodeName)});
});

app.post('/node', (req, res) => {
  res.json({graph: rust.create_node(req.body)});
});

app.put('/node', (req, res) => {
  res.json({graph: rust.modify_node(req.params.body)});
});

app.delete('/node/:nodeName', (req, res) => {
  res.json({graph: rust.delete_node(req.params.nodeName)});
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

