const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Ollama } = require('ollama');
const store = require('./store');

const ollama = new Ollama();

function readFilesRecursively(dir) {
  let results = [];

  const list = fs.readdirSync(dir);

  list.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      results = results.concat(readFilesRecursively(filePath));
    } else {
      if (
        file.endsWith('.js') ||
        file.endsWith('.md') ||
        file.endsWith('.json')
      ) {
        const content = fs.readFileSync(filePath, 'utf-8');

        results.push({
          filePath,
          content,
        });
      }
    }
  });

  return results;
}

function chunkText(text, chunkSize = 500, overlap = 50) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = start + chunkSize;
    chunks.push(text.slice(start, end));
    start += chunkSize - overlap;
  }

  return chunks;
}

async function getEmbedding(text) {
  try {
    const response = await ollama.embeddings({
      model: 'nomic-embed-text',
      prompt: text,
    });

    return response.embedding;
  } catch (error) {
    console.error('Embedding error:', error);
    return [];
  }
}

function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function saveVectorStore() {
  fs.writeFileSync('vectors.json', JSON.stringify(store.vectorStore));
}

function loadVectorStore() {
  if (fs.existsSync('vectors.json')) {
    const data = fs.readFileSync('vectors.json', 'utf-8');
    const parsed = JSON.parse(data);
    store.vectorStore = parsed;
    return store.vectorStore;
  }

  return store.vectorStore;
}

function generateChunkId(repoPath, filePath, chunk) {
  return crypto
    .createHash('md5')
    .update(repoPath + filePath + chunk)
    .digest('hex');
}

async function processChunksInParallel(
  chunks,
  filePath,
  repoPath,
  chunkStartIndex = 0,
  concurrency = 5,
  existingIds = new Set()
) {
  for (let i = 0; i < chunks.length; i += concurrency) {
    const batch = chunks.slice(i, i + concurrency);
    const batchStartIndex = chunkStartIndex + i;

    const embeddings = await Promise.all(
      batch.map((chunk) => getEmbedding(chunk))
    );

    embeddings.forEach((embedding, index) => {
      const chunkId = generateChunkId(repoPath, filePath, batch[index]);
      store.vectorStore.push({
        id: chunkId,
        repo: repoPath,
        filePath: filePath,
        chunk: batch[index],
        embedding: embedding,
      });
      existingIds.add(chunkId);
    });
  }
}

module.exports = {
  readFilesRecursively,
  chunkText,
  getEmbedding,
  cosineSimilarity,
  saveVectorStore,
  loadVectorStore,
  processChunksInParallel,
};
