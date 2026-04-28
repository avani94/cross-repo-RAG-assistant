let vectorStore = [];

app.get('/read-repo', (req, res) => {
  const repoPath = './nodejs-express-mongodb-starter-project';

  try {
    const files = readFilesRecursively(repoPath);

    vectorStore = [];

    files.forEach((file) => {
      const chunks = chunkText(file.content);

      chunks.forEach((chunk) => {
        const embedding = getEmbedding(chunk);

        vectorStore.push({
          filePath: file.filePath,
          chunk: chunk,
          embedding: embedding,
        });
      });
    });

    console.log('Stored embeddings:', vectorStore.length);

    res.json({
      message: 'Embeddings stored',
      totalChunks: vectorStore.length,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Error processing repo' });
  }
});