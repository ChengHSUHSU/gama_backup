from src_v2.configs import Config


BaseDINConfigs = Config({
    'DENSE_FEATURE_SIZE': {},
    'DENSE_FEATURE': [],

    'FEATURE_EMBEDDING_SIZE': {},
    'ONE_HOT_FEATURE': [],

    'BEHAVIOR_FEATURE': [],
    'BEHAVIOR_FEATURE_SEQ_LENGTH': [],
    'BEHAVIOR_SEQUENCE_SIZE': 10,

    'MODEL_PARAMS': {
        'dnn_use_bn': False,
        'dnn_hidden_units': (256, 128),
        'dnn_activation': 'relu',
        'att_hidden_size': (64, 16),
        'att_activation': 'Dice',
        'att_weight_normalization': False,
        'l2_reg_dnn': 0.0,
        'l2_reg_embedding': 1e-06,
        'dnn_dropout': 0,
        'init_std': 0.0001,
        'seed': 1024,
        'task': 'binary',
        'gpus': None
    }
})
