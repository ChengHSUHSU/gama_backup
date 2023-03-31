import os
import pickle
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import KFold
from google.cloud import storage

from utils.logger import logger
from utils.advanced_metrics import get_average_metrics


DAY_STRING_FORMAT = '%Y%m%d'
HOUR_STRING_FORMAT = '%Y%m%d%H'
HOUR_ONLY_STRING_FORMAT = '%H'


def read_pickle(file_name, base_path="./datasets"):
    with open(f'{base_path}/{file_name}', 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as err:
            print('fail to load pickle file.')
            return None


def dump_pickle(file_name, data, base_path='./'):

    mkdir(base_path)

    file_path = f'{base_path}/{file_name}'

    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f'Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.')
    except Exception as e:
        print(f'Downloaded storage object {source_blob_name} failed.')
        raise e


def get_norm_value(val):
    if val > 1:
        return 1 + (val - 1) / 10
    elif val < 0:
        return 1 + (val + 1) / 10
    else:
        return val


def calculate_dynamic_score(size, mean=0, std=0.5, weight=0.01):

    random_val = np.random.normal(mean, std, size)
    random_val = [get_norm_value(val) * weight for val in random_val]

    return random_val


def save_data_to_gs(df, bucket_name, result_folder, cols=None):
    tmp_file_name = 'tmp.csv'
    if cols:
        df.to_csv(tmp_file_name, header=True, columns=cols, encoding='utf-8')
    else:
        df.to_csv(tmp_file_name, encoding='utf-8')

    upload_blob(bucket_name, tmp_file_name, result_folder)


def cal_metrics(y_true, y_pred, option=[]):
    y_binary = [round(value) for value in y_pred]

    r = recall_score(y_true, y_binary, average=None)
    p = precision_score(y_true, y_binary, average=None)
    f1 = f1_score(y_true, y_binary, average=None)

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    result = (p, r, f1,)
    if 'roc' in option:
        result = result + (fpr, tpr,)
    if 'auc' in option:
        result = result + (auc(fpr, tpr),)
    return result


def metric_ndcg(df, y_pred, n=5, groupby_key='userid'):
    """
    Arguments:

    df (pd.dataframe): raw dataset, which must include ['openid', 'y']
    y_pred (nd.array): model predict result
    n (int): top-n ranking of ndcg
    groupby_key (str): ndcg primary key
    """

    df.rename(columns={'y': 'y_true'}, inplace=True)
    df.loc[:, 'y_pred'] = y_pred
    ndcg_scores = get_average_metrics(df, k=n, metric_type='ndcg', groupby_key=groupby_key)

    return ndcg_scores


def cross_validation(model_factory, X_train, y_train, k=5, random_state=None, ndcg_groupby_key='userid', ndcg='5,10,20', train_params={}, **kwargs):
    """Kfold cross validation"""

    NDCG_AT_K_LIST = ndcg.split(',')

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    metrics = {
        'precision': np.empty((kf.n_splits, 2)),
        'recall': np.empty((kf.n_splits, 2)),
        'f1': np.empty((kf.n_splits, 2)),
        'auc': np.empty((kf.n_splits, 1)),
        'roc_curve': [],
        'ndcg': {}
    }

    for i, index in enumerate(kf.split(X_train)):
        train_index, test_index = index

        logger.info(f'Train on {len(train_index)}, validate on {len(test_index)}')

        kf_X_train = X_train.iloc[train_index]
        kf_y_train = y_train.iloc[train_index]
        kf_X_test = X_train.iloc[test_index]
        kf_y_test = y_train.iloc[test_index]

        if model_factory.mode == 'train':
            model_factory.initialize(mode='train', col2label2idx={}, **kwargs)
            model_factory.train(kf_X_train, kf_y_train, train_params)

        y_pred = model_factory.predict(kf_X_test)
        p, r, f, fpr, tpr, auc = cal_metrics(kf_y_test, y_pred, option=['roc', 'auc'])
        metrics['precision'][i] = p
        metrics['recall'][i] = r
        metrics['f1'][i] = f
        metrics['auc'][i] = auc
        metrics['roc_curve'].append({'fpr': fpr, 'tpr': tpr})

        if ndcg_groupby_key:
            for ndcgk in NDCG_AT_K_LIST:
                metrics['ndcg'].setdefault(f'ndcg{ndcgk}', [])
                ndcg_score = metric_ndcg(X_train.iloc[test_index], y_pred,
                                         int(ndcgk), groupby_key=ndcg_groupby_key)
                metrics['ndcg'][f'ndcg{ndcgk}'].append(ndcg_score)

                if i == (k-1):
                    metrics['ndcg'][f'ndcg{ndcgk}'] = np.array(metrics['ndcg'][f'ndcg{ndcgk}']).mean(axis=0)

    result = (
        metrics['precision'].mean(axis=0),
        metrics['recall'].mean(axis=0),
        metrics['f1'].mean(axis=0),
        metrics['auc'].mean(axis=0),
        metrics['roc_curve'],
        metrics['ndcg'])

    return result
