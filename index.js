const express = require('express');
const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');

const app = express();
app.use(express.json());

const ollama = new Ollama();

let vectorStore = [];

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
  fs.writeFileSync(
    'vectors.json',
    JSON.stringify(vectorStore, null, 2)
  );
}

function loadVectorStore() {
  if (fs.existsSync('vectors.json')) {
    const data = fs.readFileSync('vectors.json', 'utf-8');

    vectorStore = JSON.parse(data);

    console.log(
      'Loaded vectors:',
      vectorStore.length
    );
  }
}

app.get('/', (req, res) => {
  res.send('Cross Repo AI Analyzer is running');
});

app.get('/read-repo', async (req, res) => {
  const repoPaths = [
    './nodejs-express-mongodb-starter-project',
    './express-generator-typescript',
  ];

  try {
    vectorStore = [];

    for (const repoPath of repoPaths) {
      const files = readFilesRecursively(repoPath);

      for (const file of files) {
        const chunks = chunkText(file.content);

        for (const chunk of chunks) {
          const embedding = await getEmbedding(chunk);

          vectorStore.push({
            repo: repoPath,
            filePath: file.filePath,
            chunk: chunk,
            embedding: embedding,
          });
        }
      }
    }

    console.log('Stored embeddings:', vectorStore.length);
    saveVectorStore();

    res.json({
      message: 'Embeddings stored',
      totalChunks: vectorStore.length,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      message: 'Error processing repo',
    });
  }
});

app.get('/query', async (req, res) => {
  try {
    const userQuery = req.query.q;

    if (!userQuery) {
      return res.status(400).json({
        message: 'Query parameter q is required',
      });
    }

    const queryEmbedding = await getEmbedding(userQuery);

    const scoredResults = vectorStore.map((item) => {
      const score = cosineSimilarity(
        queryEmbedding,
        item.embedding
      );

      return {
        repo: item.repo,
        filePath: item.filePath,
        chunk: item.chunk,
        score: score,
      };
    });

    scoredResults.sort((a, b) => b.score - a.score);

    const topResults = scoredResults.slice(0, 3);

    const grouped = {};

    topResults.forEach((item) => {
      if (!grouped[item.repo]) {
        grouped[item.repo] = [];
      }

      grouped[item.repo].push({
        filePath: item.filePath,
        chunk: item.chunk,
        score: item.score,
      });
    });

    res.json({
      query: userQuery,
      resultsByRepo: grouped,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      message: 'Error querying vector store',
    });
  }
});

app.get('/ask', async (req, res) => {
  try {
    const userQuery = req.query.q;

    if (!userQuery) {
      return res.status(400).json({
        message: 'Query parameter q is required',
      });
    }

    const queryEmbedding = await getEmbedding(userQuery);

    const scoredResults = vectorStore.map((item) => {
      const score = cosineSimilarity(
        queryEmbedding,
        item.embedding
      );

      return {
        repo: item.repo,
        filePath: item.filePath,
        chunk: item.chunk,
        score: score,
      };
    });

    scoredResults.sort((a, b) => b.score - a.score);

    const topResults = scoredResults.slice(0, 3);

    const context = topResults
      .map((item) => {
        return `
REPO: ${item.repo}
FILE: ${item.filePath}

CODE:
${item.chunk}
`;
      })
      .join('\n\n----------------\n\n');

    const prompt = `
You are a senior software engineer AI.

You are analyzing MULTIPLE code repositories.

Your job:
- Compare implementations across repos
- Explain differences clearly
- Mention repo names when relevant

Context:
${context}

Question:
${userQuery}
`;

    const response = await ollama.chat({
      model: 'llama3.2',
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    res.json({
      question: userQuery,
      answer: response.message.content,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      message: 'Error generating answer',
    });
  }
});

const PORT = 3000;
loadVectorStore();

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});