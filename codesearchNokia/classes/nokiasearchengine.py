from codesearch.utils import load_model
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
from codesearch.data import load_snippet_collection, load_eval_dataset, load_train_dataset

class embed_model_usedEmbed:

    def query_retrieval(self):
        query = "plot a bar chart"
        snippets = [{
            "id": "1",
            "description": "Hello world",
            "code": "print('hello world')",
            "language": "python"
            }]

        embedding_model = load_model("used-embedder-pacs")
        retrieval_model = EmbeddingRetrievalModel(embedding_model)
        retrieval_model.add_snippets(snippets)
        retrieval_model.query(query)

class model_config: 
    def __init__(self):
        self.snippets = load_snippet_collection(snippets = load_snippet_collection(collection_name))
        self.embedding_model = load_model("used-embedder-pacs")
        self.retrieval_model = EmbeddingRetrievalModel(embedding_model)
        self.retrieval_model.add_snippets(self.snippets)