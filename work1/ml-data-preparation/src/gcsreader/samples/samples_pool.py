from pyspark.sql import DataFrame

from src.gcsreader.utils import filter_candidate
from src.gcsreader.utils import extend_similar_score_field


def get_similarity_samples_pool(df: DataFrame,
                                top_k: int,
                                dedup_unique_col: list = None,
                                select_col: list = None,
                                with_col_name: str = 'similarity_score',
                                cal_embedding_col_name: list = None) -> DataFrame:
    """Function to get similarity samples pool

    Args:
        df (DataFrame)
        top_k (int): get top k candidates
        dedup_unique_col (str, optional): dedup unique col.
        select_col (list, optional): fields to display. Defaults to None.
        with_col_name (str, optional): extend similar score field name. Defaults to 'similarity_score'.
        cal_embedding_col_name (list, optional): the name of the field that needs to use embedding. Defaults to None.

    Returns:
        DataFrame
    """
    if select_col is None:
        select_col = ['userid', 'content_id', 'publish_time']
    
    if cal_embedding_col_name is None:
        cal_embedding_col_name = ['user_title_embedding', 'item_title_embedding']
    
    if dedup_unique_col is None:
        dedup_unique_col = ['userid']

    df = extend_similar_score_field(df, with_col_name, cal_embedding_col_name)
    df = filter_candidate(df, top_k, with_col_name, dedup_unique_col, select_col)

    return df
