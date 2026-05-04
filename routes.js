const express = require('express');
const fs = require('fs');
const simpleGit = require('simple-git');
const { Ollama } = require('ollama');

const store = require('./store');
const {
  readFilesRecursively,
  chunkText,
  getEmbedding,
  cosineSimilarity,
  saveVectorStore,
  processChunksInParallel,
  generateChunkId,
} = require('./utils');

const ollama = new Ollama();

const router = express.Router();

router.get('/', (req, res) => {
  res.send('Cross Repo AI Analyzer is running');
});

router.post('/load-repos', async (req, res) => {
  try {
    const repoUrls = req.body.repoUrls;

    if (!repoUrls || repoUrls.length === 0) {
      return res.status(400).json({
        message: 'repoUrls required',
      });
    }

    for (const repoUrl of repoUrls) {
      const repoName = repoUrl
        .split('/')
        .pop()
        .replace('.git', '');

      const repoPath = `./repos/${repoName}`;

      if (!fs.existsSync(repoPath)) {
        await simpleGit().clone(repoUrl, repoPath);
      } else {
      }
    }

    res.json({
      message: 'Repositories loaded successfully',
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      message: 'Error loading repositories',
    });
  }
});

router.get('/read-repo', async (req, res) => {
  const reposFolder = './repos';

  const repoPaths = fs
    .readdirSync(reposFolder)
    .map((repo) => `${reposFolder}/${repo}`);

  try {
    const existingIds = new Set(store.vectorStore.map(v => v.id));

    for (const repoPath of repoPaths) {
      const files = readFilesRecursively(repoPath);
      let chunkIndex = 0;

      for (const file of files) {
        const chunks = chunkText(file.content);
        const newChunks = [];
        const newChunkIndices = [];

        for (let i = 0; i < chunks.length; i++) {
          const chunk = chunks[i];
          const chunkId = generateChunkId(repoPath, file.filePath, chunk);
          if (existingIds.has(chunkId)) {
            continue; // skip completely
          }

          newChunks.push(chunk);
          newChunkIndices.push(chunkIndex + i);
        }

        if (newChunks.length > 0) {
          await processChunksInParallel(
            newChunks,
            file.filePath,
            repoPath,
            newChunkIndices[0],
            5,
            existingIds
          );
        }

        chunkIndex += chunks.length;
      }
    }

    saveVectorStore();

    res.json({
      message: 'Embeddings stored',
      totalChunks: store.vectorStore.length,
    });
  } catch (error) {
    console.error(error);

    res.status(500).json({
      message: 'Error processing repo',
    });
  }
});

router.get('/query', async (req, res) => {
  try {
    const userQuery = req.query.q;

    if (!userQuery) {
      return res.status(400).json({
        message: 'Query parameter q is required',
      });
    }

    const queryEmbedding = await getEmbedding(userQuery);

    const scoredResults = store.vectorStore.map((item) => {
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

router.get('/ask', async (req, res) => {
  try {
    const userQuery = req.query.q;

    if (!userQuery) {
      return res.status(400).json({
        message: 'Query parameter q is required',
      });
    }

    const queryEmbedding = await getEmbedding(userQuery);

    const scoredResults = store.vectorStore.map((item) => {
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

module.exports = router;