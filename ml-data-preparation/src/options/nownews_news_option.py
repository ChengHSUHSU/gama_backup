from .base_options import BaseOptions


class NowNewsNewsPreparationOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--enable_positive_sampling', action='store_true', default=True, help='if sample positive data')
        parser.add_argument('--daily_positive_sample_size', type=int, default=10000, help='Positive sampling option')
        parser.add_argument('--daily_sample_seed', type=int, default=1024, help='Random seed')
        parser.add_argument('--requisite_sequence_length', type=int, default=0, help='Length of requisite behavior of user')
        return parser
