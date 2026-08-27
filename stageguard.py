import numpy as np
import pandas as pd
import config as con


class Gatekeeper:
    '''
    Provide designated, preprocessed data slices to distinct model phases.
    Data provided depends on model training phase (training on fit, validating, training on train, testing) and
    on the model class (classic ML vs ICL).
    Preprocessing constants are inherited from con.PRE_PROS_CONSTANTS
    '''

    def __init__(self):
        # initialize stage parameters
        self._stage = None 
        self._previous_stage= None         
        self.model = None

        # load data and parameters
        self.panel = pd.read_parquet(con.PANEL)
        self.geo_id = pd.read_parquet(con.REF_GEOGRAPHY, columns=(['orgpermid', 'lvl3permid']))
        self.split = pd.read_parquet(con.SPLIT)
        self.pre_processing_constants = con.PRE_PROS_CONTS

        # merge data and indicators
        self.merged_data = self.panel.merge(self.geo_id, on='orgpermid', how='left')
        self.merged_data = self.merged_data.merge(self.split, on="orgpermid", how='left')

        # perform preprocessing
        self.preprocesssed_data = self.preprocessing(self.merged_data)

    def _require(self, expected, model):
        if self._stage != expected:
            raise RuntimeError(f"needs {expected}, saw {self._stage}")
        if model == None:
            raise RuntimeError(f"expects model class (ICL, nICL), recieved {model}")
        elif model != 'ICL' or model != 'nICL':
            raise RuntimeError(f"expects model class (ICL, nICL), recieved {None}")
        else:
            self.model = model

    def __call__(self, input):
        if self.model == 'nICL':
            if self.stage == 1:
                self.stage_one_preprocessing(self.data)
                return 'stage one preprocessed data'
            elif self.stage == 2:
                self.stage_two_preprocessing(self.data)
                return 'stage two preprocessed data'
            elif self.stage == 3:
                self.stage_three_preprocessing(self.data)
                return 'stage three preprocessed data'
            elif self.stage == 4:
                self.stage_four_preprocessing(self.data)
                return 'stage four preprocessed data'

    def stage_one_data(self):
        '''
        Partition: fit  
        Condition of X: imputed, scaled 
        Targets delivered: yes
        Condition of y: untouched, native [0,1] scale, NaN-free 
        Roster: non-ICL models
        Duty: Provide data for training the model on fit partition.
        Output: X (n_{partition}, 125)
        '''
        pass

    def stage_two_data(self):
        '''
        Precondition: stage 1 ran
        Partition: val   
        Condition of X: imputed, scaled  
        Targets delivered: yes 
        Condition of y: untouched, native [0,1] scale, NaN free  
        Roster: non-ICL models
        Duty: Provide data for tuning the trained models hyperparamters. Output: X (n_{val}, 125)
        '''
        
    
    def stage_three_data(self):
        '''
        Precondition: stage 2 ran == TRUE
        Partition: train
        Condition of X: scaled, imputed (inherited from PRE_PROS_CONTS)
        Targets delivered: yes
        Condition of y: untouched, native [0,1] scale, NaN free
        Roster: all models
        Duty: provide data for last training the models before scoring
        '''
        pass

    def stage_four_data(self):
        '''
        Precondition: stage 3 ran == TRUE
        Partition: test
        Condition of X: scaled, imputed (inherited from PRE_PROS_CONTS)
        Targets delivered: yes
        Condition of y: untouched, native [0,1] scale, NaN free
        Additional identifiers: orgpermid and/or lvl3permid per row
        Roster: all models
        Duty: provide data for scoring the models
        '''
        pass

    def preprocessing(self, data):
        '''
        scale and impute dataset
        '''
        preprocessed_data = data[['orgpermid', 'lvl3permid', 'partition', 'esg_combined_score']]
        for feature in self.pre_processing_constants['varible']:
            preprocessed_feature = self.merged_data[[feature, 'orgpermid']]
            feature_mean = self.pre_processing_constants.loc[self.pre_processing_constants['variable']==feature]['mean'].values[0]
            feature_median = self.pre_processing_constants.loc[self.pre_processing_constants['variable']==feature]['median'].values[0]
            feature_std = self.pre_processing_constants.loc[self.pre_processing_constants['variable']==feature]['std'].values[0]
            preprocessed_feature[feature] = preprocessed_feature[feature].fillna(feature_median)
            preprocessed_feature[feature] = (preprocessed_feature[feature] - feature_mean)/ feature_std
            preprocessed_data = preprocessed_data.merge(preprocessed_feature, on='orgpermid', how='left')
        return preprocessed_data
        

