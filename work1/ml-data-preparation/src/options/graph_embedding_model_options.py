from .base_options import BaseOptions


class GraphEmbeddingModelOptions(BaseOptions):

    def initialize(self, parser):
        parser = super(GraphEmbeddingModelOptions, self).initialize(parser)
        parser.add_argument('--logger_name', default='graph_embedding_model_data_preparation')

        return parser
