from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nmslib

class fileInputs:
    """Organizes all the necessary elements we need to make a search engine."""
    def __init__(self):
        self.url_df = pd.read_csv('data/processed_data/without_docstrings.lineage', header=None, names=['url'])
        self.code_df = pd.read_json('data/processed_data/without_docstrings_original_function.json.gz')
        code_df.columns = ['code']
        
    def downloadNew(self, urldf, codedf):
        self.url_df = pd.read_csv(urldf, header=None, names=['url'])
        self.code_df = pd.read_json(codedf)
        self.code_df.columns = ['code']
        assert code_df.shape[0] == url_df.shape[0]
        
#solid principle liskov Substitution
class customFileClass(fileInputs):
    def __init__(self):
        self.url_df = pd.read_csv('data/processed_data/without_docstrings.lineage', header=None, names=['url'])
        self.code_df = pd.read_json('data/processed_data/without_docstrings_original_function.json.gz')
        self.code_df.columns = ['code']
        
    def downloadNew(self, urldf, codedf):
        self.url_df = pd.read_csv(urldf, header=None, names=['url'])
        self.code_df = pd.read_json(codedf)
        self.code_df.columns = ['code']
        assert code_df.shape[0] == url_df.shape[0]
        
    def printHeadofURL(self):
        print(self.url_df.head(10))
    def printHeadofCODE(self):
        print(self.code_df.head(10))
    
