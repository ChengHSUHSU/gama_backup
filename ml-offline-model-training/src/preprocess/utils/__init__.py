from utils.logger import logger
from src.preprocess.utils.tag_parser import ner_parser
from src.preprocess.utils.encoder import UNKNOWN_LABEL
import numpy as np
import json
import pickle
import ast
import math
import pandas as pd
from typing import Union


def encoding(x, max_length):
    """
    Support both one-hot and multi-hot encoding
    """
    if isinstance(x, list) and (len(x) == 0):
        return np.zeros(max_length)

    x = np.array(x)
    if x.max()+1 > max_length:
        raise ValueError('label is out of max length')

    vector = np.zeros(max_length)
    vector[x] = 1
    return vector


def read_pickle(file_name, base_path='./datasets'):
    with open(f'{base_path}/{file_name}', 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as err:
            print('fail to load pickle file.')
            return None


def get_embedding_interaction(user_title: Union[list, np.array, pd.Series], item_title: Union[list, np.array, pd.Series], option='dot_product'):
    """
    Args:
        user_title (Union[list, np.array, pd.Series]): first input dataframe.
        item_title (Union[list, np.array, pd.Series]): second input dataframe.
        option (str, optional): matrix calculation method. Support dot_prod, elementwise_product, sum. Defaults to 'dot_product'.

    Raises:
        ValueError: Input Type Error.

    Returns:
        list: matrix calculation result.
    """
    try:
        if type(user_title) == pd.Series:
            user_title = np.array(user_title.values.tolist())
        if type(item_title) == pd.Series:
            item_title = np.array(item_title.values.tolist())
        if type(user_title) != pd.Series:
            user_title = np.array(user_title)
            item_title = np.array(item_title)
    except Exception as e:
        raise TypeError('Not Supporting Transform Type')

    try:
        if option == 'dot_product':
            if user_title.ndim == 1:
                result = np.dot(user_title, item_title)
            else:
                result = np.sum(user_title*item_title, axis=1)
        elif option == 'elementwise_product':
            result = user_title*item_title
        elif option == 'sum':
            result = user_title+item_title
    except Exception as e:
        raise ValueError('Get Embedding Interaction Error')
    return result.tolist()


def convert_type(dataset, mode2cols, verbose=True):
    """Function to convert col values by ast.literal_eval or json.loads (ex: embedding: '[1,2,....,0]' -> [1,2,...,0])
    """
    def _ast_convert_type(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    def _json_convert_type(x):
        # json.loads is faster for embedding
        if isinstance(x, str):
            return json.loads(x)
        return x

    for mode, cols in mode2cols.items():

        for col in cols:
            if verbose:
                logger.info(f'mode: {mode} - column: {col}')

            if mode == 'ast':
                dataset[col] = dataset[col].apply(_ast_convert_type)
            elif mode == 'json':
                dataset[col] = dataset[col].apply(_json_convert_type)
            else:
                dataset[col] = dataset[col].astype(mode)

    return dataset


def convert_unit(dataset, mode2cols):
    """Function to convert col unit
    """

    def _millisecs_convert_unit(x):
        """Function to convert milliseconds to seconds
        """
        if isinstance(x, int) or isinstance(x, float):
            return x/1000
        elif isinstance(x, str):
            return float(x)/1000
        return x

    for mode, cols in mode2cols.items():
        logger.info(f'mode: {mode}')

        for col in cols:
            logger.info(f'process: {col}')

            if mode == 'millisecs_to_secs':
                dataset[col] = dataset[col].apply(_millisecs_convert_unit)

    return dataset


def process_embedding(dataset: pd.DataFrame, embed_col1: str, embed_col2: str, dim: int = 300, mode: str = 'prod'):
    """
    Args:
        dataset (pd.DataFrame): input dataframe
        embed_col1 (str): first input column of dataframe  need to do matrix calculation.
        embed_col2 (str): second input column of dataframe  need to do matrix calculation.
        dim (int, optional): embedding dimension. Defaults to 300.
        mode (str, optional): matrix calculation mode, support 'prod', 'dot', 'sum'. Defaults to 'prod'.

    Raises:
        ValueError: Matrix calculation setting error.

    Returns:
        pd.DataFrame: final dataframe after process embedding.
    """
    mode_mapping = {
        'prod': 'elementwise_product',
        'dot': 'dot_product',
        'sum': 'sum'
    }

    if mode not in ['prod', 'dot', 'sum']:
        raise ValueError(f'Only support [`prod`, `dot`, `sum`] mode, but get {mode}')
    dataset[embed_col1] = dataset[embed_col1].apply(lambda x: x if isinstance(x, list) and len(x) == dim else [0.0] * dim)
    dataset[embed_col2] = dataset[embed_col2].apply(lambda x: x if isinstance(x, list) and len(x) == dim else [0.0] * dim)

    # process embedding
    embedding = get_embedding_interaction(dataset.loc[:, embed_col1], dataset.loc[:, embed_col2], option=mode_mapping[mode])
    dataset[f'semantics_{mode}'] = embedding

    return dataset


def parse_content_ner(dataset, cols):

    def _parse_ner(data):
        """Function to parse existence named entities
        :param data: NER tag dictionary. e.g. data = {'PERSON-伊能靜': 1.0, 'PERSON-小哈利': 5.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0}
        :type data: dict

        :returns: list of existed named entities. e.g. result = ['person', 'location']
        :rtype: list of string
        """

        existed_entities = []

        # e.g. {'person': ['伊能靜', '小哈利', '庾澄慶'], 'location': ['台北'], 'event': [], 'organization': [], 'item': [], 'others': []}
        ners = ner_parser(data)

        for entity, candidates in ners.items():

            if len(candidates) > 0:
                existed_entities.append(entity)

        return existed_entities

    for col, new_col in cols.items():
        logger.info(f'process: {col}; new column name: {new_col}')

        dataset[new_col] = dataset[col].apply(_parse_ner)

    return dataset


def cal_ner_matching(ner_1, ner_2):
    """Function to match named entities 1 and named entities 2

    :param ner_1: named entity 1. (ex: {'PERSON-伊能靜': 1.0, 'PERSON-小哈利': 5.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0})
    :type ner_1: dict
    :param ner_2: named entity 2. (ex: {'PERSON-伊能靜': 1.0, 'PERSON-小哈利': 5.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0})
    :type ner_2: dict

    :returns: matching score
    :rtype: int
    """

    if ner_1 is None or ner_2 is None:
        return 0

    score = 0
    ner_candidates = ner_1.keys()

    for entity in ner_candidates:
        if entity in ner_2:
            score += 1

    return score


def cal_features_sim(mode='cos'):

    def _cal_features_sim(feature_1, feature_2):

        if mode == 'cos':
            if sum(feature_1) == 0 or sum(feature_2) == 0:
                return 0
            return np.dot(feature_1, feature_2)/(np.linalg.norm(feature_1)*np.linalg.norm(feature_2))
        elif mode == 'dot':
            return np.dot(feature_1, feature_2)

    return _cal_features_sim


def cal_features_matching(feature_1: list, feature_2: list):
    """Function to calculate matching score

    Time cost : 0.002 ms for each feature_1-feature_2 pair
    Time cost : 0.4   ms for 200  feature_1-feature_2 pair

    :param feature_1: feature 1. (ex: ['新奇', '生活', '頭條焦點', '即時'])
    :type feature_1: list
    :param feature_2: feature 2. (ex: ['新奇', '生活'])
    :type feature_2: list

    :returns: matching score    (ex: 2)
    :rtype: int
    """

    score = 0

    if feature_1 and feature_2:
        intersection = set(feature_1) & set(feature_2)
        score = len(intersection)

    return score


def calculate_min_max(x, max_value, min_value):
    '''Function to do min-max normalization to input value

    :param x: input value
    :type x: float
    :param max_value: maximum value
    :type max_value: float
    :param min_value: minimum value
    :type min_value: float
    :return: normalized value
    :rtype: float
    '''
    normed_value = (x - min_value) / (max_value - min_value) if max_value - min_value > 0 else 0

    return normed_value


def normalize_data(dataset, cols):
    for mode in cols:
        if mode not in ['min-max', 'z-score']:
            raise ValueError(f'Only support [`min-max`, `z-score`] mode, but get {mode}')

        if mode == 'min-max':
            for col in cols[mode]:
                max = cols[mode][col]
                min = dataset[col].min()
                if max is not None:  # upper bound value
                    dataset[col] = dataset[col].apply(lambda x: 1 if calculate_min_max(x, max, min) > 1 else calculate_min_max(x, max, min))
                else:
                    dataset[col] = dataset[col].apply(lambda x: calculate_min_max(x, dataset[col].max(), min))

        if mode == 'z-score':
            for col in cols[mode]:
                dataset[col] = dataset[col].apply(lambda x: (x - dataset[col].mean()) / dataset[col].std())

    return dataset


def get_publish_time_to_now(df, event_time_col='timestamp', publish_time_col='publish_time', time_type=['min', 'hour']):
    """TODO: For computing time period, use `process_time_period` instead of this from now on
    """

    time_type_to_sec = {'min': 60, 'hour': 3600}

    for t in time_type:

        df[f'{t}_to_current'] = df.apply(
            lambda x: (x[event_time_col]-x[publish_time_col]) / (time_type_to_sec[t]*1000), axis=1)

    return df


def process_time_period(df, start_time_col='publish_time', end_time_col='timestamp', time_type=['min', 'hour'], postfix='_time_period'):
    """Function to compute time period, input `start_time` and `end_time` must be second.

    Args:
        df (pandas.Dataframe): dataframe
        start_time_col (str, optional): start time column name. Defaults to 'publish_time'.
        end_time_col (str, optional): end time column name. Defaults to 'timestamp'.
        time_type (list, optional): time period type. Defaults to ['min', 'hour'].
        postfix (str, optional): postfix of time period column. Defaults to '_time_period'.

    Returns:
        df (pandas.Dataframe)
    """

    time_type = [time_type] if isinstance(time_type, str) else time_type

    time_type_to_sec = {'min': 60, 'hour': 3600}

    for t in time_type:

        df[f'{t}{postfix}'] = df.apply(lambda x: (x[end_time_col]-x[start_time_col]) / (time_type_to_sec[t]), axis=1)

    return df


def process_age(df, columns=['age'], age_range=20):
    """Function to convert age from numerical data to categorical data
    Args:
        df (pandas.Dataframe): dataframe
        columns (list, optional): list of age column names. Defaults to 'age'.
        age_range (int, optional): range of age group. Defaults to 20.

    Returns:
        df (pandas.Dataframe)
    """

    columns = [columns] if isinstance(columns, str) else columns

    for column_name in columns:
        if column_name in df.columns:
            df[column_name] = df[column_name].apply(lambda x: str(int(x)//age_range) if not bool(checknull(x)) else UNKNOWN_LABEL)

    return df


def process_gender(df, columns=['gender'], gender_mapping={'male': 'male', 'female': 'female', 'm': 'male', 'f': 'female'}):
    """Function to convert gender to standardized data

    Args:
        df (pandas.Dataframe): dataframe
        columns (list, optional): list of gender column names. Defaults to ['gender'].
        gender_mapping (dict, optional): mapping table of gender data. Defaults to {}.

    Returns:
        df (pandas.Dataframe)
    """

    columns = [columns] if isinstance(columns, str) else columns

    for column_name in columns:
        if column_name in df.columns:
            df[column_name] = df[column_name].apply(lambda x: gender_mapping.get(x, 'other') if x is not None else UNKNOWN_LABEL)

    return df


def process_tag(df: pd.DataFrame, columns: list = ['tags']) -> pd.DataFrame:

    def ignore_entity(tags):
        res = [val.split('-')[-1] for val in tags if val.split('-')[-1] != '']
        return list(set(res))

    for column_name in columns:
        df[column_name] = df[column_name].apply(ignore_entity)

    return df


def checknull(x):

    result = False

    if (x is None) or (not bool(x)):
        result = True
    elif not isinstance(x, str):
        if (math.isnan(x)):
            result = True
        elif (np.isnan(x)):
            result = True

    return result


def merge_data_with_user(dataset: pd.DataFrame, user_data: pd.Series, configs: dict) -> pd.DataFrame:
    """Function to append user profile to input data

    Args:
        dataset (pd.DataFrame): input data
        user_data (pd.Series): user profile, can be history or realtime
        configs (dict): configs illustrate how to append user profile data.
                        Format: {profile_name: [(is_append_all_profile, new_col_name, profile_col_name), ...], ...}

    Returns:
        pd.DataFrame: data with user profile data
    """

    for (profile_name, parsing_params) in configs.items():
        user_profile = user_data.get(profile_name, {})

        for (is_append_all_profile, new_col_name, profile_col_name) in parsing_params:
            if is_append_all_profile:
                dataset[new_col_name] = json.dumps(user_profile)
            elif isinstance(user_profile.get(profile_col_name), list):
                dataset[new_col_name] = [user_profile.get(profile_col_name) for i in dataset.index]
            else:
                dataset[new_col_name] = user_profile.get(profile_col_name)

    return dataset
