import pytest
import unittest
from bdds_recommendation.src.monitoring import ModelMonitor


def test_compare_model_performance():

    init_args = {'mlflow_host': None, 'experiment_id': None, 'service_endpoint': None, 'service_type': None, 'content_type': None, 'model_type': None}

    input_metrics_current = {'val_ndcg5': 0.5, 'val_ndcg10': 0.2, 'val_ndcg20': 0.9, 'val_auc': 0.1}
    input_metrics_compared = {'val_ndcg5': 0.25, 'val_ndcg10': 0.4, 'val_ndcg20': 0.9, 'val_auc': 0.4}

    output = ModelMonitor(**init_args).compare_model_performance(metrics_current=input_metrics_current,
                                                                 metrics_compared=input_metrics_compared,
                                                                 metric_prefix='val_',
                                                                 postfix='_pir',
                                                                 requisite_metrics=['ndcg5', 'ndcg10', 'ndcg20', 'auc'])

    expect = {'ndcg5_pir': 100.0, 'ndcg10_pir': -50.0, 'ndcg20_pir': 0.0, 'auc_pir': -75.0}

    assert output == expect
