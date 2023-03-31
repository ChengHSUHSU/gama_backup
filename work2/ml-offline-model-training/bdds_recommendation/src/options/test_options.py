from bdds_recommendation.src.options.base_options import BaseOptions


class TestDINOptions(BaseOptions):
    """
    This class includes training options of DIN model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--tag', type=str, default='latest', help='date format of model tag, defult: latest')
        return parser


class TestXgboostOptions(BaseOptions):
    """
    This class includes training options of DIN model.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--tag', type=str, default='latest', help='date format of model tag, defult: latest')
        return parser
