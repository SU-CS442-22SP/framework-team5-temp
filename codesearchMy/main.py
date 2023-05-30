from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
import nmslib

from utils.lang_model_utils import load_lm_vocab, Query2Emb
from utils.general_utils import create_nmslib_search_index

from classes.readerClass import fileInputs

import classes.searchengine as searchEngine
#import classes.searchengine.search_engine as search_engine
fileReader1 = fileInputs()
f = open('data.json')
data = json.load(f)
def myFileRead():
    
    lnk1 = 'data/processed_data/without_docstrings.lineage'
    lnk2 = 'data/processed_data/without_docstrings_original_function.json.gz'
    fileReader1.downloadNew(lnk1, lnk2)
    

myFileRead()
print(fileReader1.url_df.head(10))
print(fileReader1.code_df.head(10))
# collect these two together into a dataframe
ref_df = pd.concat([fileReader1.url_df, fileReader1.code_df], axis = 1).reset_index(drop=True)
print(ref_df.head(10))    

######
#.lang_model_cpu_v2.torch
lang_model = torch.load('./data/lang_model/lang_model_cpu_v2.torch', 
                        map_location=lambda storage, loc: storage)

#.vocab_v2.cls
vocab = load_lm_vocab('./data/lang_model/vocab_v2.cls')
q2emb = Query2Emb(lang_model = lang_model.cpu(),
                  vocab = vocab)

search_index = nmslib.init(method='hnsw', space='cosinesimil')
#.search_index.nmslib
search_index.loadIndex('./data/search/search_index.nmslib')


#####
    
se = searchEngine.search_engine(nmslib_index=search_index,
           ref_df=ref_df,
           query2emb_func=q2emb.emb_mean)

#Single Responsibility Principle SOLID #1
se.search('read data into pandas dataframe')
#se.search('New Query.')

