import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from core.embeddings import embedder

from core.intents_data import intent_embeddings, intent_labels


# --------------------------------
# Predict intent
# --------------------------------
def predict_intent_semantic(user_input, threshold=0.55):

    emb = embedder.encode([user_input])

    sims = cosine_similarity(emb, intent_embeddings)[0]

    idx = np.argmax(sims)

    score = sims[idx]

    if score < threshold:
        return "unknown", score

    return intent_labels[idx], score
