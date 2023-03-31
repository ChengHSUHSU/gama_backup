import pandas as pd
import numpy as np
from typing import Union

from bdds_recommendation.utils.logger import logger
from bdds_recommendation.src_v2.configs.show.show_user2item import ShowUser2ItemDINConfig
from bdds_recommendation.src_v2.configs.show.show_user2item_sgd import ShowUser2ItemSGDConfig
from bdds_recommendation.src_v2.preprocess.show.show_user2item import ShowUser2ItemDINPreprocessor
from bdds_recommendation.src_v2.handler.din_handler import DINHandler
from bdds_recommendation.src_v2.factory import CTRRankerFactory


from bdds_recommendation.src_v2.handler.dcn_handler import DCNHandler
from bdds_recommendation.src_v2.handler.sgd_handler import SGDHandler


class ShowUser2ItemLearn2RankDINFactory(CTRRankerFactory):

    def __init__(self, mode: str = 'train', col2label2idx: dict = None, model_path: str = '/ml-model', **kwargs):

        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.mode = mode
        self.model_path = model_path
        self.initialize(self.mode, col2label2idx=col2label2idx, model_path=model_path, **kwargs)

    def initialize(self, mode: str, col2label2idx: dict = None, **kwargs):

        self.config = ShowUser2ItemDINConfig

        if mode == 'train':
            self.preprocessor = ShowUser2ItemDINPreprocessor(configs=self.config,
                                                             col2label2idx=col2label2idx,
                                                             mode=mode,
                                                             **kwargs)
            self.model_params = kwargs
        else:
            if mode == 'inference':
                self.config.REQUISITE_COLS.remove('y')

            self.preprocessor = ShowUser2ItemDINPreprocessor(configs=self.config,
                                                             col2label2idx=col2label2idx,
                                                             mode=mode)
            self.model_handler = DINHandler(col2label2idx=self.preprocessor.col2label2idx,
                                            config=self.config,
                                            mode=mode,
                                            use_cuda=False,
                                            **self.config.MODEL_PARAMS)
            self.model_handler.load_model(kwargs.get('model_path', '/ml-model/user2item/show/din/model.latest.pickle'))

    def train(self, dataset: pd.DataFrame, y_true: Union[pd.Series, np.array, list], train_params: dict = None, **kwargs):

        dataset = self.preprocessor.process(dataset=dataset,
                                            requisite_cols=kwargs.get('requisite_cols', None),
                                            **kwargs)

        self.processed_dataset = dataset
        logger.info(f'Processed data columns: {dataset.columns} \nProcessed data: \n {dataset.iloc[len(dataset) // 3]}')
        print('precessed dataset : ')
        print(dataset.head(5))
        print()

        self.model_handler = DINHandler(col2label2idx=self.preprocessor.col2label2idx,
                                        config=self.config,
                                        mode=self.mode,
                                        **self.model_params)
        self.model_handler.train(x_train=dataset, y_train=y_true, train_params=train_params)

    def predict(self, dataset, **kwargs):

        dataset = self.preprocessor.process(dataset, **kwargs)
        scores = self.model_handler.predict(dataset).reshape(-1)

        return scores


class ShowUser2ItemLearn2RankDCNFactory(CTRRankerFactory):

    def __init__(self, mode: str = 'train', col2label2idx: dict = None, model_path: str = '/ml-model', **kwargs):

        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.mode = mode
        self.processed_dataset = None
        self.processed_dataset_pred = None
        self.col2label2idx = col2label2idx
        self.initialize(self.mode, col2label2idx=col2label2idx, model_path=model_path, **kwargs)

    def initialize(self, mode: str, col2label2idx: dict = None, **kwargs):

        self.config = ShowUser2ItemDINConfig

        if mode == 'train':
            self.preprocessor = ShowUser2ItemDINPreprocessor(configs=self.config,
                                                             col2label2idx=col2label2idx,
                                                             mode=mode,
                                                             **kwargs)
            self.model_params = kwargs
        else:
            if mode == 'inference':
                self.config.REQUISITE_COLS.remove('y')

            self.preprocessor = ShowUser2ItemDINPreprocessor(configs=self.config,
                                                             col2label2idx=col2label2idx,
                                                             mode=mode)
            self.model_handler = DCNHandler(col2label2idx=self.preprocessor.col2label2idx,
                                            config=self.config,
                                            mode=mode,
                                            use_cuda=False,
                                            **self.config.MODEL_PARAMS)
            self.model_handler.load_model(kwargs.get('model_path', '/ml-model/user2item/show/dcn/model.latest.pickle'))

    def train(self, dataset: pd.DataFrame, y_true: Union[pd.Series, np.array, list], train_params: dict = None, **kwargs):
        dataset = self.preprocessor.process(dataset=dataset,
                                            requisite_cols=kwargs.get('requisite_cols', None),
                                            **kwargs)
        self.processed_dataset = dataset
        
        logger.info(f'Processed data columns: {dataset.columns} \nProcessed data: \n {dataset.iloc[len(dataset) // 3]}')

        self.model_handler = DCNHandler(col2label2idx=self.preprocessor.col2label2idx,
                                        config=self.config,
                                        mode=self.mode,
                                        **self.model_params)
        self.model_handler.train(x_train=self.processed_dataset, y_train=y_true, train_params=train_params)

    def predict(self, dataset, **kwargs):
        dataset = self.preprocessor.process(dataset, **kwargs)
        scores = self.model_handler.predict(dataset).reshape(-1)

        return scores


class ShowUser2ItemLearn2RankSGDFactory(CTRRankerFactory):

    def __init__(self, mode: str = 'train', col2label2idx: dict = None, model_path: str = '/ml-model', **kwargs):

        assert mode in self.MODE, 'only support train, validation and inference mode'
        self.mode = mode
        self.model_path = model_path
        self.initialize(self.mode, col2label2idx=col2label2idx, model_path=model_path, **kwargs)

    def initialize(self, mode: str, col2label2idx: dict = None, **kwargs):

        self.config = ShowUser2ItemSGDConfig

        if mode == 'train':
            self.preprocessor = ShowUser2ItemDINPreprocessor(configs=self.config,
                                                             col2label2idx=col2label2idx,
                                                             mode=mode,
                                                             **kwargs)
            self.model_params = kwargs
        else:
            if mode == 'inference':
                self.config.REQUISITE_COLS.remove('y')

            self.preprocessor = ShowUser2ItemDINPreprocessor(configs=self.config,
                                                             col2label2idx=col2label2idx,
                                                             mode=mode)
            self.model_handler = SGDHandler(config=self.config,
                                            mode=self.mode,
                                            **self.config.MODEL_PARAMS)
    
            self.model_handler.load_model(kwargs.get('model_path', '/ml-model/user2item/show/sgd/model.latest.pickle'))

    def train(self, dataset: pd.DataFrame, y_true: Union[pd.Series, np.array, list], train_params: dict = None, **kwargs):

        dataset = self.preprocessor.process(dataset=dataset,
                                            requisite_cols=kwargs.get('requisite_cols', None),
                                            **kwargs)

        self.processed_dataset = dataset
        logger.info(f'Processed data columns: {dataset.columns} \nProcessed data: \n {dataset.iloc[len(dataset) // 3]}')

        self.model_handler = SGDHandler(config=self.config,
                                        mode=self.mode,
                                        **self.config.MODEL_PARAMS)
        print('dataset.columns : ', dataset.columns)
        '''
        dataset.columns :  Index(['gender', 'age', 'cat0', 'hist_cat0', 'seq_length_cat0',
       'user_title_embedding', 'item_title_embedding', 'cat0_pref_score',
       'hour_time_period', 'popularity_score', '1_last_day_total_view_count',
       '1_last_day_total_click_count', '3_last_day_total_view_count',
       '3_last_day_total_click_count', '5_last_day_total_view_count',
       '5_last_day_total_click_count', 'y'],
        '''
        print('y_true : ', y_true)

        self.model_handler.train(x_train=dataset, y_train=y_true, train_params=train_params)

    def predict(self, dataset, **kwargs):

        dataset = self.preprocessor.process(dataset, **kwargs)
        scores = self.model_handler.predict(dataset).reshape(-1)

        return scores
