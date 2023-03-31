from .base_options import BaseOptions


class PostUser2ItemOptions(BaseOptions):

    def initialize(self, parser):
        """Define user2item arguments that are used in data preparation."""
        parser = super(PostUser2ItemOptions, self).initialize(parser)
        parser.add_argument('--logger_name', default='club_post_user2item_data_preparation')
        parser.add_argument('--daily_positive_sample_size', type=int, default=3000, help='Positive sampling option')
        parser.add_argument('--daily_sample_seed', type=int, default=1024, help='Random seed')

        return parser
