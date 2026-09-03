import numpy as np
import pandas as pd
import config as con

class Gatekeeper():
    '''
    Provide designated, preprocessed data slices to distinct model phases.
    Data provided depends on model training phase (training on fit, validating, training on train, testing) and
    on the model class (in-Context-Learing (ICL) vs. non-ICL (nICL)).
    Preprocessing constants are inherited from con.PRE_PROS_CONSTANTS
    '''

### define initializers  --------------------------------------------------------- 

    def __init__(self, model):
        print(f'\n\n[+][+][+] intializing data gatekeeper [+][+][+]\n\n')
        # initialize stage parameters
        self.model = model
        self._stage = None   

        assert self.model in {'ICL', 'nICL'}, f"expected model classes: ICL or nICL, received {self.model}"

        # load data and parameters
        self.panel = pd.read_parquet(con.PANEL)
        self.geo_id = pd.read_parquet(con.REF_GEOGRAPHY, columns=(['orgpermid', 'lvl3permid']))
        self.split = pd.read_parquet(con.SPLIT)
        self.pre_processing_constants = pd.read_parquet(con.PRE_PROS_CONTS)

        # merge data and indicators
        self.merged_data = self.panel.merge(self.geo_id, on='orgpermid', how='left')
        self.merged_data = self.merged_data.merge(self.split, on="orgpermid", how='left')

        print(f"excluding observations from tier 3 regions")
        l_before = len(self.merged_data)
        self.regions = [*con.TIER1_REGS, *con.TIER2_REGS]
        self.merged_data = self.merged_data.loc[self.merged_data['lvl3permid'].isin(self.regions)]
        l_after = len(self.merged_data)
        print(f"excluded observations: {l_before-l_after}")

        # perform preprocessing
        self._preprocessed_data = self._preprocessing()

        # print intitalization convergence
        print(f'\n \n [-] data gatekeeper initialized for model class {self.model} [-] \n [-] proceed with stage call [-]\n \n')

    def _preprocessing(self):
        '''
        Scale and impute dataset.
        '''
        feature_names = self.pre_processing_constants['variable']
        variables_to_keep = [*feature_names, 'esg_combined_score', 'orgpermid', 'lvl3permid', 'partition', ]
        pre_processed_data = self.merged_data[variables_to_keep].copy()
        constants = self.pre_processing_constants.set_index('variable')
        print(f"    [>>>] initializing data preprocessing - raw data shape: {self.merged_data.shape}")
        for feature in constants.index: 
            feature_mean = constants.loc[feature, 'mean']
            feature_median = constants.loc[feature, 'median']
            feature_std = constants.loc[feature, 'std']
            pre_processed_data[feature] = pre_processed_data[feature].fillna(feature_median)
            pre_processed_data[feature] = (pre_processed_data[feature] - feature_mean)/ feature_std
        print(f"    [>>>] data preprocessing succesfull - data shape:  {pre_processed_data.shape} ")
        return pre_processed_data

    def _require(self, expected):
        if self._stage not in expected:
                raise RuntimeError(f"needs stage {expected}, saw {self._stage}")

    def _require_class(self, expected):
        if self.model not in expected:
            raise RuntimeError(f"needs model class {expected}, saw {self.model}")

    def _slice_data(self, partitions, stage):
        stage_data = self._preprocessed_data[self._preprocessed_data['partition'].isin(partitions)]
        if stage in {None,1,2,3}:
            geo_ID = None
        elif stage in {4}:
            geo_ID = stage_data['lvl3permid']
        stage_data = stage_data.drop(columns=['orgpermid', 'partition', 'lvl3permid'])
        stage_X = stage_data.drop(columns=['esg_combined_score'])
        stage_y = stage_data['esg_combined_score']
        return stage_X, stage_y, geo_ID

### define methods ---------------------------------------------------------

    def stage_one_data(self):
        '''
        Output data for stage 1 model phase.
        Output:
            - partition: fit  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: stage == None OR previous_stage == 2, model == nICL
        '''

        # check conditions
        self._require({None, 2})
        self._require_class({'nICL'})

        # set stage parameter
        self._stage = 1

        # slice data
        X,y,_ = self._slice_data({'fit'}, self._stage)
        assert X.shape[0] == y.shape[0], f"stage 1 X-dimensions and y-dimension mismatch"
        print(f"stage 1 data provided, X - {X.shape}, y- {y.shape}")
        return X, y
### ---------------------------------------------------------

    def stage_two_data(self):
        '''
        Output data for stage 2 model phase.
        Output:
            - partition: val  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 1, model class == 'nICL' 
        '''

        # check conditions
        self._require({1})
        self._require_class({'nICL'})

        # set stage parameter
        self._stage = 2

        # slice data
        X,y,_ = self._slice_data({'val'}, self._stage)
        assert X.shape[0] == y.shape[0], f"stage 2 X-dimensions and y-dimension mismatch"
        print(f"stage 2 data provided, X - {X.shape}, y - {y.shape}")
        return X, y
### ---------------------------------------------------------

    def stage_three_data(self):
        '''
        Output data for stage 3 model phase.
        Output:
            - partition: train (fit+val)  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 2, model class == 'nICL' 
        '''

        # check conditions
        if self.model== 'nICL':
            self._require({2})
        elif self.model=='ICL':
            self._require({None})

        # set stage parameter
        self._stage = 3

        # slice data
        X,y,_ = self._slice_data({'fit', 'val'}, self._stage)
        assert X.shape[0] == y.shape[0], f"stage 3 X-dimensions and y-dimension mismatch"
        print(f"stage 3 data provided, X - {X.shape}, y - {y.shape}")
        return X, y
### ---------------------------------------------------------

    def stage_four_data(self):
        '''
        Output data for stage 4 model phase.
        Output:
            - partition: test  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 3 
        '''
        # check conditions (no class condition required)
        self._require({3})

        # set stage parameter
        self._stage = 4

        # slice data
        X,y,geo_ID = self._slice_data({'test'}, self._stage)
        assert X.shape[0] == y.shape[0], f"stage 4 X-dimensions and y-dimension mismatch"
        print(f"stage 4 data provided, X - {X.shape}, y - {y.shape}, geo-ID - {geo_ID.shape}")
        return X, y, geo_ID
### ---------------------------------------------------------



        

