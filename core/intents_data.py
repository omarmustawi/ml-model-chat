from core.embeddings import embedder
from constants.keywords import generic_templates

# --------------------------------
# Intent examples
# --------------------------------
intent_texts = []
intent_labels = []

for intent, examples in generic_templates.items():

    for ex in examples:

        intent_texts.append(ex)
        intent_labels.append(intent)

# --------------------------------
# Intent embeddings
# --------------------------------
intent_embeddings = embedder.encode(intent_texts)
