from .base_options import BaseOptions


class NewsItem2ItemOptions(BaseOptions):

    def initialize(self, parser):
        """Define item2item arguments that are used in data preparation."""
        parser = super(NewsItem2ItemOptions, self).initialize(parser)
        parser.add_argument('--logger_name', default='planet_news_item2item_data_preparation')

        return parser


class NewsUser2ItemOptions(BaseOptions):

    def initialize(self, parser):
        parser = super(NewsUser2ItemOptions, self).initialize(parser)
        parser.add_argument('--logger_name', type=str, default='planet_news_user2item_data_preparation')
        parser.add_argument('--service_type', type=str, default='user2item')

        parser.add_argument('--requisite_sequence_length', type=int, default=0, help='Length of requisite behavior of user')
        parser.add_argument('--daily_positive_sample_size', type=int, default=10000, help='Positive sampling option')
        parser.add_argument('--daily_sample_seed', type=int, default=1024, help='Random seed')

        parser.add_argument('--topk_freshness', type=int, default=100)
        parser.add_argument('--topk_popular', type=int, default=100)
        parser.add_argument('--topk_similar', type=int, default=100)

        parser.add_argument('--similar_sample_ratio', type=float, default=0.2, help='Similar positive sample ratio, range [0.0, 1.0]')
        parser.add_argument('--similar_sample_size', type=int, default=20, help='Similar negative sample size')
        parser.add_argument('--random_sample_size', type=int, default=20, help='Random negative sample ratio')

        return parser

class NewsHotOptions(BaseOptions):
    def initialize(self, parser):
        """Define hot items arguments that are used in data preparation."""
        parser = super(NewsHotOptions, self).initialize(parser)
        parser.add_argument('--item_content_type', type=str, default='others')
        parser.add_argument('--logger_name', default='planet_news_hot_item_data_preparation')
        parser.add_argument('--input_bucket', type=str, default='event-bf-data-uat-001')

        return parser
