from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
import nmslib
import unittest

from classes.readerClass import fileInputs
from classes.readerClass import customFileClass

class myFileRead(unittest.TestCase):
    fileReader1 = fileInputs()
    lnk1 = 'data/processed_data/without_docstrings.lineage'
    lnk2 = 'data/processed_data/without_docstrings_original_function.json.gz'
    fileReader1.downloadNew(lnk1, lnk2)

class customClassUsage(unittest.TestCase):
    customReader = customFileClass()
    customReader.printHeadofURL()
    lnk1 = 'data/processed_data/without_docstrings.lineage'
    lnk2 = 'data/processed_data/without_docstrings_original_function.json.gz'
    customReader.downloadNew(lnk1, lnk2) # solid principle liskov, can still download with its own method rather than parent

if __name__ == '__main__':
    unittest.main()
