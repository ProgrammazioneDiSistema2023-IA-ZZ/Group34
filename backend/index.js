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
  res.json({graph: rust.select_model(req.params.ordinal)});
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

