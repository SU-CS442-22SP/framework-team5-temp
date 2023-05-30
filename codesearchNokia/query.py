# © 2020 Nokia
#
# Licensed under the BSD 3 Clause license
#
# SPDX-License-Identifier: BSD-3-Clause
# ============================================

import unittest
import json

from codesearch.utils import load_model
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
from codesearch.data import load_snippet_collection, load_eval_dataset, load_train_dataset

#f = open('without_docstrings_original_function.json')
#snippets = json.dumps('without_docstrings_original_function.json')

query = "plot a bar chart"
#snippets = load_snippet_collection("so-ahmet-framework-data")
snippets = [{
    "id": "1",
    "description": "Hello world",
    "code": "print('hello world')",
    "language": "python"
    }]

embedding_model = load_model("use-embedder-pacs")
retrieval_model = EmbeddingRetrievalModel(embedding_model)
retrieval_model.add_snippets(snippets)
print(retrieval_model.query(query))

#https://fasttext.cc/docs/en/support.html