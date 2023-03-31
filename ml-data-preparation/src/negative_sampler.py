from tqdm import tqdm
import pandas as pd
import random
import traceback
import logging


def _negative_sampler(df, sample_size, major_col='openid', candidate_col='uuid', time_col='publish_time',
                      enable_distinct_major=True, at_least_appear_k=0, random_choice_retry=3):
    """
    Function for negative pair generation
    Scenario : keep negative sample published before positive sample.
    """
    def _random_select_sample(df):
        sample_row = df.iloc[random.randint(0, len(df)-1)]  # df.iloc[X] is much faster than df.sample()
        publish_time = sample_row[time_col]
        candidate_data = sample_row[candidate_col]
        return candidate_data, publish_time

    RANDOM_CHOICE_RETRY = random_choice_retry
    negative_samples = {}

    if enable_distinct_major:
        # generate from distinct major data
        major_col_set = list(set(df[major_col]))
    else:
        major_col_set = list(df[major_col])

    df[time_col] = df[time_col].astype(int)

    if at_least_appear_k:
        df_appear_count = df.groupby([candidate_col]).size().to_frame('count')
        df = df.merge(df_appear_count, how='left', on=candidate_col)
        df = df[df['count'] >= at_least_appear_k]

    for major_col_data in tqdm(major_col_set, desc='[negative sampling]'):
        try:
            major_data = df[df[major_col] == major_col_data]    # 慢在這邊
            candidate_data = list(major_data[candidate_col])

            # latest candidate major data co-occur
            latest_timestamp = major_data.loc[major_data[time_col].idxmax()]
            latest_timestamp = int(latest_timestamp[time_col])

            samples = []

            for _ in range(sample_size):
                current_choice_count = 0
                item, this_timestamp = _random_select_sample(df)

                condition = (
                    item in candidate_data or
                    item in samples or
                    this_timestamp > latest_timestamp
                )

                while condition:

                    item, this_timestamp = _random_select_sample(df)

                    if current_choice_count > RANDOM_CHOICE_RETRY:
                        break
                    else:
                        current_choice_count += 1

                if not condition:
                    samples.append(item)

            negative_samples[major_col_data] = negative_samples.get(major_col_data, []) + samples
        except Exception as e:
            logging.info(traceback.format_exc())
    return negative_samples


def generate_negative_samples(df, major_col='openid', candidate_col='uuid', time_col='publish_time', sample_size=20, enable_distinct_major=True, prefix=''):
    """
    Function for generating negative dataframe
    """
    if prefix == '' and major_col == candidate_col:
        raise ValueError(f'`prefix` is needed if `major_col` is equal to `candidate_col`')

    negative_samples = _negative_sampler(df, sample_size, major_col, candidate_col, time_col, enable_distinct_major)

    # Generate negative pair dataframe
    df_neg_pair = pd.DataFrame(columns=[major_col, prefix + candidate_col])
    for major_data, candidate_data in negative_samples.items():
        df = pd.DataFrame(data={major_col: major_data, prefix + candidate_col: candidate_data})
        df_neg_pair = df_neg_pair.append(df, ignore_index=True)

    return df_neg_pair
