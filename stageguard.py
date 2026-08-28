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

    def __init__(self):
        # initialize stage parameters
        self._stage = None        
        self.model = None

        # load data and parameters
        self.panel = pd.read_parquet(con.PANEL)
        self.geo_id = pd.read_parquet(con.REF_GEOGRAPHY, columns=(['orgpermid', 'lvl3permid']))
        self.split = pd.read_parquet(con.SPLIT)
        self.pre_processing_constants = pd.read_parquet(con.PRE_PROS_CONTS)

        # merge data and indicators
        self.merged_data = self.panel.merge(self.geo_id, on='orgpermid', how='left')
        self.merged_data = self.merged_data.merge(self.split, on="orgpermid", how='left')

        # perform preprocessing
        self.preprocessed_data = self.preprocessing(self.merged_data)

    def _require(self, expected):
        if self._stage != expected:
            raise RuntimeError(f"needs {expected}, saw {self._stage}")

    def _check_model(self, model):
        if model.isin(['ICL', 'nICL']):
            self.model = model
        else:
            raise RuntimeError(f"expects model class (ICL, nICL), recieved {None}")

    def __call__(self, model):
        '''Initialize class on model type and current stage:
            [*] possible model types: "ICL", "nICL"
            [*] stages: 1, 2, 3, 4:
                [**] stage 1:
                    - condition: none 
                    - partition: fit
                    - available for model types: nICL 
                    - output:
                        - X (n,125), preprocessed.
                        - y (n,1), native in [0,1], NaN-free (by pull design)
                    
                [**] stage 2: 
                    - condition: stage 1 ran before 
                    - partition: val
                    - available for model types: nICL
                    output:
                        - X (n,125), preprocessed.
                        - y (n,1), native in [0,1], NaN-free (by pull design)
                        
                [**] stage 3:
                    - condition: stage 2 ran before
                    - partition: train (fit+val)
                    - available for model types: nICL & ICL
                    -output:
                        - X (n,125), preprocessed.
                        - y (n,1), native in [0,1], NaN-free (by pull design)
                                                        
                [**] stage 4:
                    - condition: stage 3 ran before
                    - partition: test
                    - available for model types: nICL & ICL
                    - output    
                        - X (n,125), preprocessed.
                        - y (n,1), native in [0,1], NaN-free (by pull design)
                        - geographic region identifier (lvl3permid)
        '''
        self._check_model(self.model)

        if self.model == 'nICL':
            if self._stage == 1:
                return self.stage_one_preprocessing(self.preprocessed_data)
            
            elif self._stage == 2:
                return self.stage_two_preprocessing(self.preprocessed_data)
            
            elif self._stage == 3:
                return self.stage_three_preprocessing(self.preprocessed_data)

            elif self._stage == 4:
                 return self.stage_four_preprocessing(self.preprocessed_data)
        
        elif self.model == 'ICL':
            if self._stage == 1:
                raise RuntimeError(f"needs stage {None}, saw {self._stage}")
            
            elif self._stage == 2:
                raise RuntimeError(f"needs stage {None}, saw {self._stage}")
            
            elif self._stage == 3:
                raise RuntimeError(f"needs stage {None}, saw {self._stage}")

            elif self._stage == None:
                self._stage = 3
                return self.stage_three_preprocessing(self.preprocessed_data)

            elif self._stage == 4:
                return self.stage_four_preprocessing(self.preprocessed_data)

        

    def stage_one_data(self):
        '''
        Output data for stage 1 model phase.
        Output:
            - partition: fit  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Condition: stage == None OR previous_stage == 2
        '''
        if self.stage == None:
            self._require(None)
        elif self.stage == (2):
            self._require(2)
        stage_data = self.preprocessed_data.query('partition==fit')
        stage_data = stage_data(columns=['orgpermid', 'lvl3permid', 'partition'])
        self.stage = 1
        print(f"stage 1 data provided - (n,X+y)= {len(stage_data)}")
        return stage_data

    def stage_two_data(self):
        '''
        Output data for stage 2 model phase.
        Output:
            - partition: val  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 1 
        '''
        self._require(1)
        stage_data = stage_data.query('partition==val')
        stage_data = self.preprocessed_data.drop(columns=['orgpermid', 'lvl3permid', 'partition'])
        self.stage = 2
        print(f"stage 2 data provided - (n,X+y)= {len(stage_data)}")
        return stage_data
        
    def stage_three_data(self):
        '''
        Output data for stage 3 model phase.
        Output:
            - partition: train (fit+val)  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 2 
        '''
        self._require(2)
        stage_data = stage_data.query('partition==val | partition==fit')
        stage_data = self.preprocessed_data.drop(columns=['orgpermid', 'lvl3permid', 'partition'])
        self.stage = 3
        print(f"stage 3 data provided - (n,X+y)= {len(stage_data)}")
        return stage_data

    def stage_four_data(self):
        '''
        Output data for stage 3 model phase.
        Output:
            - partition: train (fit+val)  
            - X (n,125), preprocessed.
            - y (n,1), native in [0,1], NaN-free (by pull design)
        Conditions: previous_stage == 3 
        '''
        self._require(3)
        stage_data = stage_data.query('partition==test')
        stage_data = self.preprocessed_data.drop(columns=['lvl3permid'])
        self.stage = 4
        print(f"stage 4 data provided - (n,X+y+geo_id)= {len(stage_data)}")
        return stage_data

    def preprocessing(self, data):
        '''
        Scale and impute dataset.
        '''
        preprocessed_features = []
        preprocessed_data = data[['orgpermid', 'lvl3permid', 'partition', 'esg_combined_score']]
        for feature in self.pre_processing_constants['variable']:
            preprocessed_feature = self.merged_data[[feature, 'orgpermid']]
            feature_mean = self.pre_processing_constants.loc[self.pre_processing_constants['variable']==feature]['mean'].values[0]
            feature_median = self.pre_processing_constants.loc[self.pre_processing_constants['variable']==feature]['median'].values[0]
            feature_std = self.pre_processing_constants.loc[self.pre_processing_constants['variable']==feature]['std'].values[0]
            preprocessed_feature[feature] = preprocessed_feature[feature].fillna(feature_median)
            preprocessed_feature[feature] = (preprocessed_feature[feature] - feature_mean)/ feature_std
            preprocessed_features.append(pd.DataFrame(preprocessed_feature))
        preprocessed_data.merge(preprocessed_features, on='orgpermid', how='left')
        return preprocessed_data
        

