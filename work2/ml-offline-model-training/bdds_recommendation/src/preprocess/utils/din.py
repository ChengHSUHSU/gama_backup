import json
import numpy as np
import pandas as pd


# TODO: Refactor this part next iteration
# 1. If we want to use the historical sequence as a feature, the user profile must be added before the event
# 2. `process_user_behavior_sequence` should not drop the current feature from the sequence
class Behavior():
    def __init__(self, sequence_size=10, padding_token=0):
        self.SEQUENCE_SIZE = sequence_size
        self.PAD_TOKEN = padding_token

    def process_hist(self, behavior, hist_list, recursive_pop_out=False):
        """Pop out exist uuid to avoid peeking"""

        if not (isinstance(hist_list, list) or isinstance(hist_list, np.ndarray)):

            raise TypeError(f'hist_list only support type of `list` and `np.array`, but get {type(hist_list)}')

        elif isinstance(hist_list, np.ndarray):
            hist_list = hist_list.tolist()

        if behavior in hist_list:

            index = hist_list.index(behavior)
            hist_list.pop(index)

            if recursive_pop_out:
                self.process_hist(behavior, hist_list)

        return hist_list

    def padding(self, data):
        """only support post padding
        """
        if self.SEQUENCE_SIZE < len(data):
            return np.array(data[-self.SEQUENCE_SIZE:])

        res = np.pad(np.array(data),
                     pad_width=[0, self.SEQUENCE_SIZE-len(data)],
                     mode='constant',
                     constant_values=self.PAD_TOKEN)
        return res

    def get_behavior_sequence_len(self, x):
        ctr = self.SEQUENCE_SIZE
        for val in x:
            if val == self.PAD_TOKEN:
                ctr -= 1
        return ctr


def process_user_behavior_sequence(dataset, encoders, major_col='openid', col='cat1', behavior_sequence_size=10, pad_label='PAD'):
    """The process support `cat1` user behavior sequence for now
    """

    COLUMN_NAME = col
    HIST_COLUMN = f'hist_{col}'
    PAD_TOKEN = encoders[COLUMN_NAME].data2index[pad_label]
    behavior_helper = Behavior(sequence_size=behavior_sequence_size, padding_token=PAD_TOKEN)

    df_positive = dataset[dataset['y'] == 1]

    df_hist = df_positive.sort_values(['publish_time'], ascending=True) \
        .groupby(major_col)[COLUMN_NAME].apply(list) \
        .reset_index(name=HIST_COLUMN)

    df_hist[HIST_COLUMN] = df_hist[HIST_COLUMN].apply(lambda x: behavior_helper.padding(x))

    dataset = dataset.merge(df_hist, on=major_col, how='left')

    # fillna with [pad_label * behavior_sequence_size]
    dataset[HIST_COLUMN] = dataset[HIST_COLUMN].apply(
        lambda x: [PAD_TOKEN for _ in range(behavior_sequence_size)] if not isinstance(x, np.ndarray) else x)

    dataset[HIST_COLUMN] = dataset.apply(
        lambda x: behavior_helper.padding(behavior_helper.process_hist(x[COLUMN_NAME], x[HIST_COLUMN].copy()))
        if x['y'] == 1 else x[HIST_COLUMN], axis=1)

    dataset['seq_length'] = dataset[HIST_COLUMN].apply(lambda x: behavior_helper.get_behavior_sequence_len(x))

    return dataset
    # TODO: Refactor this part next iteration


def padding(data: list, max_sequence_length: int = 10, pad_token: int = 0):
    """Function for padding list to `max_sequence_length` with `pad_token`

    Args:
        data (list)
        max_sequence_length (int, optional): maximum sequence length. Defaults to 10.
        pad_token (int, optional): pad_token. Defaults to 0.

    Returns:
        np.array
    """

    if max_sequence_length < len(data):
        return np.array(data[:max_sequence_length])

    data = np.pad(np.array(data),
                  pad_width=[0, max_sequence_length-len(data)],
                  mode='constant',
                  constant_values=pad_token)
    return data


def get_topk_preference_data(data: str, profile_key: str = 'click', sequence_key: str = 'cat1', max_sequence_length: int = 10):
    """Function to parse top k preference data from user profile.
    Support user profiles with the format of {<profile_key>: <sequence_key>: {id_1: {pref: xxx}}}

    Args:
        data (str)
        profile_key (str, optional): preference profile key. Defaults to 'click'.
        sequence_key (str, optional): preference sequence key. Defaults to 'cat1'.
        max_sequence_length (int, optional): maximum sequence length. Defaults to 10.

    Returns:
        list
    """
    try:
        result = json.loads(data).get(profile_key, {}).get(sequence_key, [])
    except json.JSONDecodeError:
        result = []

    if not result:
        return result

    key2pref = {k: v['pref'] if 'pref' in v else 0 for k, v in result.items()}
    key2pref = sorted(key2pref.items(), key=lambda kv: kv[1], reverse=True)
    sequence_length = len(key2pref)

    max_sequence_length = sequence_length if sequence_length < max_sequence_length else max_sequence_length
    result = [key2pref[i][0] for i in range(max_sequence_length)]
    return result


def get_behavior_feature(dataset: pd.DataFrame,
                         encoder: dict,
                         encoder_key: str = 'cat1',
                         behavior_col: str = 'user_category',
                         seq_length_col: str = 'seq_length',
                         prefix: str = 'hist_',
                         profile_key: str = 'click',
                         sequence_key: str = 'cat1',
                         max_sequence_length: int = 10,
                         mode: str = 'serving'):
    """Function to get DIN behavior feature from preference profile

    Args:
        dataset (pd.DataFrame): pandas dataframe
        encoder (dict): feature encoder
        encoder_key (str, optional): encoder key. Defaults to 'cat1'.
        behavior_col (str, optional): target column for behavior sequence. Defaults to 'user_category'.
        seq_length_col (str, optional): sequence length column name. Defaults to 'seq_length'.
        prefix (str, optional): behavior sequence column prefix. Defaults to 'hist_'.
        profile_key (str, optional): preference profile key. Defaults to 'click'.
        sequence_key (str, optional): preference sequence key. Defaults to 'cat1'.
        max_sequence_length (int, optional): maximum sequence length. Defaults to 10.
        mode (str, optional): process mode, support `training` and `serving`. Defaults to 'training'.

    Returns:
        (training) pandas dataframe
        (serving) list
    """

    pad_token = encoder.get('PAD', None)
    unknown_token = encoder.get('UNKNOWN', None)

    if pad_token is None:
        raise KeyError('The PAD_LABEL need to be set in encoder to generate sequence data')

    if unknown_token is None:
        raise KeyError('The UNKNOWN_LABEL need to be set in encoder to generate sequence data')

    if mode == 'training':

        hist_col_name = f'{prefix}{encoder_key}'

        dataset[hist_col_name] = dataset[behavior_col].apply(lambda x: get_topk_preference_data(x,
                                                                                                profile_key=profile_key,
                                                                                                sequence_key=sequence_key,
                                                                                                max_sequence_length=max_sequence_length))

        # TODO: To keep training/serving logics align, move feature `seq_length_col` creation and `hist_col_name` padding out next iteration
        dataset[seq_length_col] = dataset[hist_col_name].apply(len)
        dataset[hist_col_name] = dataset[hist_col_name].apply(lambda hist_list: [encoder.get(val, unknown_token) for val in hist_list])
        dataset[hist_col_name] = dataset[hist_col_name].apply(lambda hist_list: padding(hist_list, max_sequence_length, pad_token=pad_token))

    elif mode == 'serving':

        dataset = get_topk_preference_data(dataset,
                                           profile_key=profile_key,
                                           sequence_key=sequence_key,
                                           max_sequence_length=max_sequence_length)

        dataset = [encoder.get(val, unknown_token) for val in dataset]

    return dataset
