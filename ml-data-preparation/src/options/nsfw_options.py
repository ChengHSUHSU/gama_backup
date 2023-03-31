from .base_options import BaseOptions


class NSFWTextOptions(BaseOptions):

    def initialize(self, parser):
        parser = super(NSFWTextOptions, self).initialize(parser)
        parser.add_argument('--logger_name', default='nsfw_text_preparation')
        return parser
