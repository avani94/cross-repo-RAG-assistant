import os
os.environ["HF_HUB_TIMEOUT"] = "60"
from sentence_transformers import SentenceTransformer
import sys
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

text = sys.argv[1]

embedding = model.encode(text).tolist()

print(json.dumps(embedding))