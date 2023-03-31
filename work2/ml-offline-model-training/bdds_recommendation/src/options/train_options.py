from bdds_recommendation.src.options.base_options import BaseOptions


class TrainDINOptions(BaseOptions):
    """
    This class includes training options of DIN model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--dnn_use_bn', type=bool, default=False, help='Whether use BatchNormalization before activation or not in deep net')
        parser.add_argument('--dnn_activation', type=str, default='relu', help='Activation function to use in deep net')
        parser.add_argument('--att_activation', type=str, default='Dice', help='Activation function to use in attention net')
        parser.add_argument('--att_weight_normalization', type=bool, default=False, help='initial learning rate for adam')
        parser.add_argument('--l2_reg_dnn', type=float, default=0, help='L2 regularizer strength applied to DNN')
        parser.add_argument('--l2_reg_embedding', type=float, default=0.000001, help='L2 regularizer strength applied to embedding vector')
        parser.add_argument('--dnn_dropout', type=float, default=0, help='float in [0,1), the probability we will drop out a given DNN coordinate.')
        parser.add_argument('--init_std', type=float, default=0.0001, help='to use as the initialize std of embedding vector')
        parser.add_argument('--task', type=str, default='binary', help='``"binary"`` for  binary logloss or  ``"regression"`` for regression loss')
        parser.add_argument('--optimizer', type=str, default='adagrad', help='support `sgd`, `adam`, `adagrad`, `rmsprop`')
        parser.add_argument('--objective_function', type=str, default='binary_crossentropy', help='support `binary_crossentropy`, `mse`, `mae`')
        parser.add_argument('--metrics', type=list, nargs='*', default=['binary_crossentropy'], help='support `binary_crossentropy`, `auc`, `mse`')

        return parser


class TrainDCNOptions(BaseOptions):
    """
    This class includes training options of DCN model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--dnn_use_bn', type=bool, default=False, help='Whether use BatchNormalization before activation or not in deep net')
        parser.add_argument('--l2_reg_dnn', type=float, default=0, help='L2 regularizer strength applied to DNN')
        parser.add_argument('--l2_reg_embedding', type=float, default=0.000001, help='L2 regularizer strength applied to embedding vector')
        parser.add_argument('--dnn_dropout', type=float, default=0, help='float in [0,1), the probability we will drop out a given DNN coordinate.')
        parser.add_argument('--init_std', type=float, default=0.0001, help='to use as the initialize std of embedding vector')
        parser.add_argument('--task', type=str, default='binary', help='``"binary"`` for  binary logloss or  ``"regression"`` for regression loss')
        parser.add_argument('--optimizer', type=str, default='adagrad', help='support `sgd`, `adam`, `adagrad`, `rmsprop`')
        parser.add_argument('--objective_function', type=str, default='binary_crossentropy', help='support `binary_crossentropy`, `mse`, `mae`')
        parser.add_argument('--metrics', type=list, nargs='*', default=['binary_crossentropy'], help='support `binary_crossentropy`, `auc`, `mse`')
        return parser


class TrainLGBMOptions(BaseOptions):
    """
    This class includes training options of LGBM model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--n_estimators', type=int, default=200, help='')
        parser.add_argument('--learning_rate', type=float, default=0.1, help='')
        parser.add_argument('--objective', type=str, default='binary', help='')
        parser.add_argument('--test_size', type=float, default=0.33, help='')
        parser.add_argument('--random_state', type=int, default=42, help='')

        return parser


class TrainXGBOptions(BaseOptions):
    """
    This class includes training options of xgboost model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--num_boost_round', type=int, default=10, help='')
        parser.add_argument('--booster', type=str, default='gbtree', help='')
        parser.add_argument('--max_depth', type=int, default=100, help='')
        parser.add_argument('--eta', type=float, default=0.07, help='')
        parser.add_argument('--min_child_weight', type=int, default=1, help='')
        parser.add_argument('--objective', type=str, default='binary:logistic', help='')
        parser.add_argument('--nthread', type=int, default=4, help='')
        parser.add_argument('--missing', type=int, default=0, help='')
        parser.add_argument('--n_estimators', type=int, default=100, help='')
        parser.add_argument('--gamma', type=int, default=0, help='')
        parser.add_argument('--learning_rate', type=float, default=0.1, help='')
        parser.add_argument('--verbosity', type=int, default=2, help='0: silent, 1: warning, 2: info, 3: debug')
        parser.add_argument('--eval_metric', nargs='*', default=['auc', 'map', 'ndcg', 'logloss'])

        return parser


class TrainSGDOptions(BaseOptions):
    """
    This class includes training options of sklearn SGDClassifier model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--loss', type=str, default='log',
                            help='support `hinge`, `log`, `modified_huber`, `squared_hinge`, `perceptron`, `squared_error`, `huber`, `epsilon_insensitive`, or `squared_epsilon_insensitive`')
        parser.add_argument('--penalty', type=str, default='l2', help='support `l2`, `l1` and `elasticnet`')
        parser.add_argument('--alpha', type=float, default=0.0001, help='')
        parser.add_argument('--fit_intercept', type=bool, default=True, help='')
        parser.add_argument('--max_iter', type=int, default=1000, help='')
        parser.add_argument('--tol', type=float, default=0.001, help='')
        parser.add_argument('--n_iter_no_change', type=int, default=5, help='')
        parser.add_argument('--early_stopping', type=bool, default=False, help='')
        parser.add_argument('--shuffle', type=bool, default=True, help='')
        parser.add_argument('--learning_rate_mode', type=str, default='optimal', help='support: `constant`, `optimal`, `invscaling` and `adaptive`')
        parser.add_argument('--learning_rate', type=float, default=0.0,
                            help='The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules')
        parser.add_argument('--eval_metric', nargs='*', default=['auc', 'map', 'ndcg', 'logloss'])

        return parser


class TrainContextualBandit(BaseOptions):
    """
    This class includes training options of bandit hot model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--alpha', type=float, default=0.001, help='')
        parser.add_argument('--n_visits', type=int, default=0, help='')
        parser.add_argument('--warmup_iters', type=int, default=1, help='number of iterations to warmup')
        parser.add_argument('--shuffle', type=bool, default=True, help='')
        parser.add_argument('--run_date', default='', type=str, help='run date by hour ex. 2021050101')
        parser.add_argument('--output_blob', default='metrics/popularity/jollybuy_goods/RUN_DATE/RUN_HOUR/popularity_jollybuy_goods.csv',
                            type=str, help='metrics result output blob')

        return parser
