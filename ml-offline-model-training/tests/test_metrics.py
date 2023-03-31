import pytest
import pandas as pd


def test_get_average_metrics():
    from utils.advanced_metrics import get_average_metrics

    y_pred = [0.16299451, 0.13258384, 0.82888178, 0.99387236, 0.03496364,
              0.40896308, 0.56295183, 0.34091324, 0.64859835, 0.00303123,
              0.23333534, 0.36626483, 0.53282833, 0.16179217, 0.9175406,
              0.01295669, 0.42440756, 0.65844648, 0.80733447, 0.46145495]
    y_true = [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    userid = ['userid_2', 'userid_1', 'userid_1', 'userid_0', 'userid_3', 'userid_1',
              'userid_0', 'userid_0', 'userid_1', 'userid_1', 'userid_1', 'userid_3',
              'userid_2', 'userid_0', 'userid_0', 'userid_0', 'userid_1', 'userid_3',
              'userid_3', 'userid_3']

    df = pd.DataFrame(data={'userid': userid, 'y_pred': y_pred, 'y_true': y_true})

    # Test case1: NDCG@k
    ndcg_at5 = get_average_metrics(df, k=5, metric_type='ndcg', groupby_key='userid')
    ndcg_at10 = get_average_metrics(df, k=10, metric_type='ndcg', groupby_key='userid')

    assert ndcg_at5 == 0.734742145447694
    assert ndcg_at10 == 0.7649449711225675

    # Test case2: precision@k
    precision_at5 = get_average_metrics(df, k=5, metric_type='precision', groupby_key='userid')
    assert list(precision_at5) == [0.375, 0.35]

    precision_at10 = get_average_metrics(df, k=10, metric_type='precision', groupby_key='userid')
    assert list(precision_at10) == [0.4375, 0.35]

    # Test case3: recall@k
    precision_at5 = get_average_metrics(df, k=5, metric_type='recall', groupby_key='userid')
    assert list(precision_at5) == [0.25, 0.625]

    precision_at10 = get_average_metrics(df, k=10, metric_type='recall', groupby_key='userid')
    assert list(precision_at10) == [0.31666666666666665, 0.5416666666666666]
