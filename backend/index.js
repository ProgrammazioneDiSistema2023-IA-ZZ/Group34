const express = require('express');
const cors = require('cors');
const app = express();
const port = 3001;

//get rust functions
const rust = require('.');
console.log(rust.hello());

app.use(cors());

app.get('/get', (req, res) => {
  res.json({ message: 'Hello from Node.js backend!  && ' +  rust.hello()});
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});