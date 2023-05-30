from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nmslib
from lang_model_utils import load_lm_vocab, Query2Emb

class mytoolConfig:
    def __init__(self):
        self.url_df = pd.read_csv(r'C:\Users\Administrator\Desktop\codesearchBACKUP\data\processed_data\without_docstrings.lineage', header=None, names=['url'])
        self.code_df = pd.read_json(r'C:\Users\Administrator\Desktop\codesearchBACKUP\data\processed_data\without_docstrings_original_function.json.gz')
        self.code_df.columns = ['code']
        self.vocab = load_lm_vocab(r'C:\Users\Administrator\Desktop\codesearchBACKUP\data\lang_model\vocab_v2.cls')
        self.lang_model = torch.load(r'C:\Users\Administrator\Desktop\codesearchBACKUP\data\lang_model\lang_model_cpu_v2.torch', map_location=lambda storage, loc: storage)

class mytoolSearchEngine:
    """Organizes all the necessary elements we need to make a search engine."""
    def __init__(self, 
                 nmslib_index, 
                 ref_df, 
                 query2emb_func):
        
  
        assert 'url' in ref_df.columns
        assert 'code' in ref_df.columns

        
        self.search_index = nmslib_index
        self.ref_df = ref_df
        self.query2emb_func = query2emb_func
    
    def search(self, str_search, k=2):
        """
        Prints the code that are the nearest neighbors (by cosine distance)
        to the search query.
        
        Parameters
        ==========
        str_search : str
            a search query.  Ex: "read data into pandas dataframe"
        k : int
            the number of nearest neighbors to return.  Defaults to 2.
        
        """
        query = self.query2emb_func(str_search)
        idxs, dists = self.search_index.knnQuery(query, k=k)
        
        for idx, dist in zip(idxs, dists):
            code = self.ref_df.iloc[idx].code
            url = self.ref_df.iloc[idx].url
            print(f'cosine dist:{dist:.4f}  url: {url}\n---------------\n')
            print(code)