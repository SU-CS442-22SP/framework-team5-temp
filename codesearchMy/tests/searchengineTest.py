from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
import nmslib
from utils.lang_model_utils import load_lm_vocab, Query2Emb
from utils.general_utils import create_nmslib_search_index

class testPathsAvailability:
    input_path = Path('../data/processed_data/')
    code2emb_path = Path('../data/code2emb/')
    output_path = Path('../data/search')
    output_path.mkdir(exist_ok=True)

class testModelLoad(unittest.TestCase):
    lang_model = torch.load('../data/lang_model/lang_model_cpu_v2.torch', map_location=lambda storage, loc: storage)
    vocab = load_lm_vocab('../data/lang_model/vocab_v2.cls')
    q2emb = Query2Emb(lang_model = lang_model.cpu(), vocab = vocab)

class loadNmslib(unittest.TestCase):
    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.loadIndex('../data/search/search_index.nmslib')
    

if __name__ == '__main__':
    unittest.main()