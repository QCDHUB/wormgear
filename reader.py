import sys
import pandas as pd
from qcdlib.aux import AUX
from tools.reader import _READER
from tools.config import conf

class READER(_READER):
  
    def __init__(self):
        pass
  
    def get_idx(self,tab):
        tab['idx']=pd.Series(tab.index,index=tab.index)
        return tab
  
    def modify_table(self,tab):
        tab=self.apply_cuts(tab)
        #tab=self.get_idx(tab)
        return tab
  
