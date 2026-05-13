from sentence_transformers import SentenceTransformer
from core.config import MODEL_NAME

embedder = SentenceTransformer(MODEL_NAME)