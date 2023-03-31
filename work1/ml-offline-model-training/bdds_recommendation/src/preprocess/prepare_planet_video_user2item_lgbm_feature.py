from bdds_recommendation.src.preprocess.utils import process_embedding, read_pickle
from bdds_recommendation.src.preprocess.utils.encoder import CategoricalEncoder
from bdds_recommendation.src.preprocess.utils.preference import Preference
from bdds_recommendation.src.preprocess.utils import convert_type
from bdds_recommendation.utils.logger import logger
import pandas as pd


class PlanetVideosUser2ItemDataPreprocessor():
    """The preprocessor support `lgbm` model for now"""

    def __init__(self, opt, configs, encoders=None, logger=logger):

        self.opt = opt
        self.configs = configs
        self.logger = logger
        self.encoders = encoders if encoders else {}

    def process(self, dataset=None, requisite_cols=None):

        if isinstance(dataset, pd.DataFrame) and len(dataset):
            dataset = dataset
        else:
            dataset = self._read_data(dataset)

        # convert data type
        dataset = self._convert_data_type(dataset)

        # convert columns name
        dataset = self._convert_columns_name(dataset)

        # handle missing data
        dataset = self._handle_missing_data(dataset)

        # process embedding
        dataset = self._process_embedding(dataset, result_col_prefix=['video', 'news'])

        # aggregate preference
        dataset = self._aggregate_preference(dataset)

        # process categorical features
        dataset = self._encode_features(dataset)

        return dataset

    def _read_data(self, dataset=None):
        self.logger.info(f'---read data---')

        if self.opt.is_train:
            dataset = read_pickle(self.opt.dataset, base_path=self.opt.dataroot)

        return dataset

    def _convert_columns_name(self, dataset):
        self.logger.info(f'---convert columns name---')

        dataset = dataset.rename(columns=self.configs.COLUMN_TO_RENAME)

        return dataset

    def _handle_missing_data(self, dataset):
        self.logger.info(f'---handle missing data---')

        df_tmp = dataset[dataset['age'].notnull()]
        df_tmp['age'] = df_tmp.age.astype(int)
        median_age = int(df_tmp.age.median())

        dataset['age'] = dataset.age.fillna(median_age)
        dataset['age'] = dataset.age.apply(lambda x: median_age if int(x) < 0 else int(x))
        dataset['gender'] = dataset.gender.fillna(CategoricalEncoder.UNKNOWN_LABEL)

        return dataset

    def _convert_data_type(self, dataset):
        self.logger.info(f'---convert data type---')

        dataset = convert_type(dataset, self.configs.TYPE_CONVERT_MODE2COLS)

        return dataset

    def _process_embedding(self, dataset, result_col_prefix=[]):
        self.logger.info(f'---process embedding data---')
        prefix_idx = 0
        for feature_pair in self.configs.SEMANTICS_FEATURE:
            major_col = feature_pair[0]
            minor_col = feature_pair[1]

            for mode in self.configs.SEMANTICS_INTERACTION_MODE:
                self.logger.info(f'process : {mode}')
                dataset = process_embedding(dataset, major_col, minor_col, mode=mode)
                if result_col_prefix[prefix_idx] != '':
                    dataset = dataset.rename(
                        columns={f'semantics_{mode}': f'{result_col_prefix[prefix_idx]}_semantics_{mode}'})
            prefix_idx += 1

        return dataset

    def _aggregate_preference(self, dataset):
        self.logger.info(f'---Aggregate preference---')

        user_pref_helper = Preference(self.configs)

        self.logger.info(f'process cat1')
        dataset = user_pref_helper.extract_category_pref_score(dataset, cat_col='user_video_category', score_col='video_category_pref_score')
        dataset = user_pref_helper.extract_category_pref_score(dataset, cat_col='user_news_category', score_col='news_category_pref_score')

        self.logger.info(f'process tags')
        dataset['tags'] = dataset['tags'].apply(lambda x: user_pref_helper._parse_tags_from_string(x) if x else [])

        if self.opt.is_train:
            enable_single_user = False
        else:
            enable_single_user = True

        self.logger.info(f'process tags: editor')
        dataset = user_pref_helper.extract_tag_pref_score(dataset, tagging_type='editor', enable_single_user=enable_single_user)

        self.logger.info(f'process tags: ner')
        dataset = user_pref_helper.extract_tag_pref_score(dataset, tagging_type='ner', enable_single_user=enable_single_user)

        return dataset

    def _encode_features(self, dataset, prefix='', suffix=''):
        self.logger.info('---process categorical data---')
        data_encoder = CategoricalEncoder(col2label2idx={})

        # encode categorical features
        self.logger.info('Load categorical feature mapping dictionary')
        all_cats_file = getattr(self.configs, 'COL2CATS_NAMES', 'col2label.pickle')
        all_cats = read_pickle(all_cats_file, base_path=self.opt.dataroot)

        for feature_col, (enable_padding, enable_unknown, mode) in self.configs.CATEGORY_FEATURES_PROCESS.items():
            self.logger.info(feature_col)
            col = feature_col.replace(prefix, '').replace(suffix, '')
            dataset[feature_col] = data_encoder.encode_transform(dataset[feature_col], feature_col, all_cats[col],
                                                                 enable_padding, enable_unknown, mode)

        self.encoder = data_encoder

        return dataset
