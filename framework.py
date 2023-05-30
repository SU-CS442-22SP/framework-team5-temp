from abc import ABC, abstractmethod
from codesearchMy.utils.lang_model_utils import load_lm_vocab, Query2Emb
from codesearchMy.classes.searchengine import mytoolSearchEngine, mytoolConfig
#from codesearchNokia.nokiasearchengine import model_config

import nmslib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nmslib

class searchEngine(ABC):
    def __init__ (self, nmsIndex):
        self.name = "Search Engine Framework"
        self.nmsIndex = nmsIndex
        
    @abstractmethod
    def search(self):
        pass

class myTool(searchEngine):
    def __init__(self, nmsIndex):
        super().__init__(nmslib.init(method='hnsw', space='cosinesimil'))
        self.nmsIndex.loadIndex('./codesearchMy/data/search/search_index.nmslib')
        self.config = mytoolConfig()
        self.q2emb = Query2Emb(lang_model = self.config.lang_model.cpu(), vocab = self.config.vocab)
        self.ref_df = pd.concat([self.config.url_df, self.config.code_df], axis = 1).reset_index(drop=True)
        
        self.engine = mytoolSearchEngine(nmslib_index=self.nmsIndex, ref_df=self.ref_df, query2emb_func=self.q2emb.emb_mean)
    
    
    def search(self, query):
        self.engine.search(query)
        
class nokiaSearchTool(searchEngine):
    def __init__(self, retrieval_model):
        self.retrieval = retrieval_model
    def search(self, query):
        retrieval_model.query(query)
        
        
queryS = input("Enter query to search: ")
        
tools = [myTool(nmslib.init(method='hnsw', space='cosinesimil'))]  #nokiaSearchTool()
for tool in tools:
    tool.search(queryS)
    
    
