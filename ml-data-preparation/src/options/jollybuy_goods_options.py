from .base_options import BaseOptions


class GoodsItem2ItemOptions(BaseOptions):

    def initialize(self, parser):
        """Define item2item arguments that are used in data preparation."""
        parser = super(GoodsItem2ItemOptions, self).initialize(parser)
        parser.add_argument('--logger_name', default='jollybuy_goods_item2item_data_preparation')
        parser.add_argument('--candidate_sample_ratio', default=1,
                            help='the ratio of rows in candidate df used to generate negative data')

        return parser


class GoodsUser2ItemOptions(BaseOptions):
    def initialize(self, parser):
        """Define user2item arguments that are used in data preparation."""
        parser = super(GoodsUser2ItemOptions, self).initialize(parser)
        parser.add_argument('--logger_name', default='jollybuy_goods_user2item_data_preparation')
        parser.add_argument('--enable_positive_sampling', action='store_true', default=False, help='if sample positive data')
        parser.add_argument('--daily_positive_sample_size', type=int, default=5000, help='Positive sampling option')
        parser.add_argument('--daily_sample_seed', type=int, default=1024, help='Random seed')
        parser.add_argument('--requisite_sequence_length', type=int, default=0, help='Length of requisite behavior of user')
        parser.add_argument('--content_negative_sample_size', type=int, default=0)
        parser.add_argument('--pop_negative_sample_size', type=int, default=0)

        return parser


class GoodsHotOptions(BaseOptions):
    def initialize(self, parser):
        """Define hot items arguments that are used in data preparation."""
        parser = super(GoodsHotOptions, self).initialize(parser)
        parser.add_argument('--item_content_type', type=str, default='others')
        parser.add_argument('--logger_name', default='jollybuy_goods_hot_item_data_preparation')
        parser.add_argument('--input_bucket', type=str, default='event-bf-data-uat-001')

        return parser
