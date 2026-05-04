const express = require('express');
const { Ollama } = require('ollama');
const store = require('./store');
const { loadVectorStore } = require('./utils');
const routes = require('./routes');

const app = express();
app.use(express.json());
app.use('/', routes);

const PORT = 3000;
store.vectorStore = loadVectorStore() || [];

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});