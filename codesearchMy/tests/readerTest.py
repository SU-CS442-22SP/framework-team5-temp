from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nmslib
import unittest

class testFileAccess(unittest.TestCase):
    # Returns True or False.
    def readDocstrings(self):
        # read file of urls
        url_df = pd.read_csv('processed_data/without_docstrings.lineage', header=None, names=['url'])

        # read original code
        code_df = pd.read_json('processed_data/without_docstrings_original_function.json.gz')
        code_df.columns = ['code']
        assert code_df.shape[0] == url_df.shape[0] # make sure these files have same number of rows
        # collect these two together into a dataframe
        ref_df = pd.concat([url_df, code_df], axis = 1).reset_index(drop=True)
        ref_df.head()


if __name__ == '__main__':
    unittest.main()