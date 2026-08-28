import numpy as np
import pandas as pd
import config as con

class Gatekeeper:
    '''
    Provide designated, preprocessed data slices to distinct model phases.
    Data provided depends on model training phase (training on fit, validating, training on train, testing) and
    on the model class (in-Context-Learing (ICL) vs. non-ICL (nICL)).
    Preprocessing constants are inherited from con.PRE_PROS_CONSTANTS
    '''

### define initializers  --------------------------------------------------------- 

    def __init__(self):
        # initialize stage parameters
        print(f'\n\n[+][+][+] intializing data gatekeeper [+][+][+]\n\n')
        self._stage = None        

        # load data and parameters
        self.panel = pd.read_parquet(con.PANEL)
        self.geo_id = pd.read_parquet(con.REF_GEOGRAPHY, columns=(['orgpermid', 'lvl3permid']))
        self.split = pd.read_parquet(con.SPLIT)
        self.pre_processing_constants = pd.read_parquet(con.PRE_PROS_CONTS)

        # merge data and indicators
        self.merged_data = self.panel.merge(self.geo_id, on='orgpermid', how='left')
        self.merged_data = self.merged_data.merge(self.split, on="orgpermid", how='left')

        # perform preprocessing
        self._preprocessed_data = self._preprocessing()

        # print intitalization convergence
        print(f'\n \n [-] data gatekeeper initialized - proceed with stage call [-]\n \n')

    def _preprocessing(self):
        '''
        Scale and impute dataset.
        '''
        feature_names = self.pre_processing_constants['variable']
        variables_to_keep = [*feature_names, 'esg_combined_score', 'lvl3permid', 'partition']
        pre_processed_data = self.merged_data[variables_to_keep].copy()
        constants = self.pre_processing_constants.set_index('variable')
        print(f"initializing data preprocessing - raw data shape: {self.merged_data.shape}")
        for feature in constants.index:
            feature_mean = constants.loc[feature, 'mean']
            feature_median = constants.loc[feature, 'median']
            feature_std = constants.loc[feature, 'std']
            pre_processed_data[feature] = pre_processed_data[feature].fillna(feature_median)
            pre_processed_data[feature] = (pre_processed_data[feature] - feature_mean)/ feature_std
        print(f"data preprocessing succesfull - data shape:  {pre_processed_data.shape} ")
        return pre_processed_data

    def _require(self, expected):
        if self._stage != expected:
            raise RuntimeError(f"needs {expected}, saw {self._stage}")

### define methods ---------------------------------------------------------

    def stage_one_data(self, model, stage = 0):
        '''
        Output data for stage 1 model phase.
        Output:
            - partition: fit  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: stage == None OR previous_stage == 2, model == nICL
        '''
        # initialize
        if stage == 0:
            stage = self._stage

        # check conditions
        assert model == 'nICL', f"expected model class: nICL, received {model}"
        assert stage == None or stage == 2, f"expected stages: None OR 2, received: {stage}"

        # slice data
        stage_data = self.preprocessed_data.query('partition==fit')
        stage_data = stage_data.drop(columns=['orgpermid', 'lvl3permid', 'partition'])
        self._stage = 1
        print(f"stage 1 data provided - (n,X+y)= {len(stage_data)}")
        return stage_data

### ---------------------------------------------------------

    def stage_two_data(self, model, stage = 0):
        '''
        Output data for stage 2 model phase.
        Output:
            - partition: val  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 1, model class == 'nICL' 
        '''
                # initialize
        if stage == 0:
            stage = self._stage

        # check conditions
        assert model == 'nICL', f"expected model class: nICL, received {model}"
        self._require(1)

        # slice data
        stage_data = stage_data.query('partition==val')
        stage_data = self.preprocessed_data.drop(columns=['orgpermid', 'lvl3permid', 'partition'])
        self._stage = 2
        print(f"stage 2 data provided - (n,X+y)= {len(stage_data)}")
        return stage_data

### ---------------------------------------------------------

    def stage_three_data(self, model, stage=0):
        '''
        Output data for stage 3 model phase.
        Output:
            - partition: train (fit+val)  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 2, model class == 'nICL' 
        '''
        #initialize
        if stage == 0:
            stage = self._stage

        # check conditions
        assert model == 'ICL' or model == 'nICL', f"expected model class ICL or nICL, received {model}"
        self._require(2)

        # slice data
        stage_data = stage_data.query('partition==val | partition==fit')
        stage_data = self.preprocessed_data.drop(columns=['orgpermid', 'lvl3permid', 'partition'])
        self._stage = 3
        print(f"stage 3 data provided - (n,X+y)= {len(stage_data)}")
        return stage_data

### ---------------------------------------------------------

    def stage_four_data(self, model, stage=0):
        '''
        Output data for stage 4 model phase.
        Output:
            - partition: tset  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 3 
        '''
        #initialize
        if stage == 0:
            stage = self._stage

        # check conditions
        assert model == 'ICL' or model == 'nICL', f"expected model class ICL or nICL, received {model}"
        self._require(3)

        # slice data
        stage_data = stage_data.query('partition==test')
        stage_data = self.preprocessed_data.drop(columns=['orgpermid', 'partition'])
        self._stage = 4
        print(f"stage 4 data provided - (n,X+y+geo_id)= {len(stage_data)}")
        return stage_data

### ---------------------------------------------------------



        

