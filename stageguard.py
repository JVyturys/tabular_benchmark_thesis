##################################################
'''

'''
##################################################
import numpy as np
import pandas as pd
import config as con


class Gatekeeper:
    '''
    Provide designated, preprocessed data slices to distinct model phases.
    Data provided depends on stage of model training phase (training on fit, validating, training on train, testing) and
    on the model (classic ML vs ICL).
    Preprocessing constants are inherited from con.PRE_PROS_CONSTANTS
    
    '''

    def __init__(self):
        self._stage = None          # nothing has run
        self.panel = pd.read_parquet(con.PANEL)
        self.panel.drop(columns=['year'])
        self.geo_id = pd.read_parquet(con.REF_GEOGRAPHY, columns=(['orgpermid', 'lvl3permid']))
        self.split = pd.read_parquet(con.SPLIT)
        self.pre_prosessing_constants = con.PRE_PROS_CONTS

    def _require(self, expected):
        if self._stage != expected:
            raise RuntimeError(f"needs {expected}, saw {self._stage}")

    def stage_one_preprocessing():
        pass

    def stage_two_preprocessing():
        pass
    
    def stage_three_preprocessing():
        pass

    def stage_four_preprocessing():
        pass