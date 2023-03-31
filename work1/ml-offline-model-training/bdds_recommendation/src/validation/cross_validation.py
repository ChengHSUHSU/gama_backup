import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import KFold

from bdds_recommendation.utils.advanced_metrics import get_average_metrics, get_overall_metrics
from bdds_recommendation.src_v2.factory import CTRRankerFactory


# TODO:
# Add customized precision@k and recall@k function
def cross_validation(model_factory: CTRRankerFactory,
                     x_train: pd.DataFrame,
                     y_train: pd.DataFrame,
                     k_fold: int = 5,
                     metric_type_list: list = ['precision', 'recall', 'f1', 'auc', 'roc_curve'],
                     metric_at_k_list: list = ['ndcg'],  # After customized precision@k, recall@k function finished,
                     # add 'precision', 'recall' param in this list
                     random_state: int = None,
                     groupby_key: str = 'userid',
                     metric_at_k: str = '5,10,20',
                     train_params: dict = {},
                     **kwargs):
    """Kflod cross validation

    Args:
        model (pytorch model): Input model which want to use cross validation. ONLY SUPPORT PYTORCH MODEL NOW.
        X_train (pd.DataFrame): DataFrame of all_dataset.
        y_train (pd.DataFrame): DataFrame of ground truth column.
        k_fold (int, optional): split dataset to k folds. Defaults to 5.
        metric_type_list (list): Monitoring metrics in list. Default to precision, recall, f1, auc, roc_curve, and ndcg.
        metric_at_k_list (list): Monitoring metrics@K in in this list. ONLY SUPPORTING NDCG NOW, PRECISION and RECALL NEED TO CUSTOMIZED.
        random_state (int, optional): random_state in sklearn. Defaults to None.
        groupby_key (str, optional): _description_. Defaults to 'userid'.
        metric_at_k (str, optional): metrics @ k. Defaults to '5,10,20'.
        train_params (dict, optional): model training parameters

    Returns:
        tuple: metric result from param metric_type_list and param metric_at_k_list.
    """
    print('k_fold : ', k_fold)

    metric_at_k = metric_at_k.split(',') if isinstance(metric_at_k, str) else metric_at_k

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    metrics = {
        'precision': np.empty((kf.n_splits, 2)),
        'recall': np.empty((kf.n_splits, 2)),
        'f1': np.empty((kf.n_splits, 2)),
        'auc': np.empty((kf.n_splits, 1)),
        'roc_curve': [],
        'ndcg': {},
        'top_k_precision': {},
        'top_k_recall': {}
    }

    for i, index in enumerate(kf.split(x_train)):
        train_index, test_index = index

        kf_x_train = x_train.iloc[train_index]
        kf_y_train = y_train.iloc[train_index]

        kf_x_test = x_train.iloc[test_index]
        kf_y_test = y_train.iloc[test_index]

        if model_factory.mode == 'train':
            model_factory.initialize(mode='train', col2label2idx={}, **kwargs)
            model_factory.train(kf_x_train, kf_y_train, train_params)

        y_pred = model_factory.predict(kf_x_test)
        over_all_metrics = get_overall_metrics(kf_y_test, y_pred, metric_type_list)

        for metric_type, metric_val in over_all_metrics.items():

            if metric_type == 'roc_curve':
                metrics[metric_type].append((metric_val))
            else:
                metrics[metric_type][i] = metric_val

        # Get @k metrics
        kf_x_test['y_pred'] = y_pred
        kf_x_test = kf_x_test.rename(columns={'y': 'y_true'})

        for metric_type in metric_at_k_list:
            if isinstance(metric_at_k, int):
                metrics[metric_type] = get_average_metrics(df=kf_x_test,
                                                           k=metric_at_k,
                                                           metric_type=metric_type,
                                                           groupby_key=groupby_key)
            else:
                for at_k in metric_at_k:
                    if metric_type != 'ndcg':
                        metric_string = f'top_k_{metric_type}'
                    else:
                        metric_string = metric_type

                    metrics[metric_string].setdefault(f'{metric_type}{at_k}', [])
                    metrics[metric_string][f'{metric_type}{at_k}'].append(get_average_metrics(df=kf_x_test,
                                                                                              k=int(at_k),
                                                                                              metric_type=metric_type,
                                                                                              groupby_key=groupby_key))
                    # last round of k fold
                    if i == (k_fold-1):
                        metrics[metric_string][f'{metric_type}{at_k}'] = np.array(metrics[metric_string][f'{metric_type}{at_k}']).mean(axis=0)

    metric_results = []
    for metric_type in metric_type_list:
        if metric_type != 'roc_curve':
            metric_results.append(metrics[metric_type].mean(axis=0))
        else:
            metric_results.append(metrics[metric_type])

    for metric_type in metric_at_k_list:
        metric_results.append(metrics[metric_type])

    return tuple(metric_results)
