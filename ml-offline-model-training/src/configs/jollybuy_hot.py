from . import BaseConfig


class JollybuyGoodsContextualBanditConfig(BaseConfig):

    # only support `ast` and `json` for now
    # {mode: [col1, col2, ...]}
    TYPE_CONVERT_MODE2COLS = {'int': ['hour', 'price', 'count_total_sales', 'count_total_browse', 'count_total_favorite', 'count_comment'],
                              'ast': ['cat0']}

    # categorical columns
    COLUMNS_TO_ENCODE = ['cat0']

    # minmax normalization columns
    COLUMNS_TO_MINMAX = ['jb_merchant_content_impression_impression_cnt', 'jb_goods_content_impression_impression_cnt',
                         'jb_cart_content_impression_impression_cnt', 'jb_category_content_impression_impression_cnt',
                         'jb_search_content_impression_impression_cnt', 'jb_home_page_content_impression_impression_cnt',
                         'jb_merchant_content_click_click_cnt', 'jb_goods_content_click_click_cnt',
                         'jb_category_content_click_click_cnt', 'jb_search_content_click_click_cnt',
                         'jb_home_page_content_click_click_cnt', 'jb_goods_item_click_booster_cnt',
                         'price', 'count_total_sales', 'count_total_browse', 'count_total_favorite', 'count_comment', 'hour']

    # final columns to be used for training
    REQUISITE_COLS = ['uuid', 'uuids', 'reward', 'hour',
                      'jb_merchant_content_impression_impression_cnt', 'jb_goods_content_impression_impression_cnt',
                      'jb_cart_content_impression_impression_cnt', 'jb_category_content_impression_impression_cnt',
                      'jb_search_content_impression_impression_cnt', 'jb_home_page_content_impression_impression_cnt',
                      'jb_merchant_content_click_click_cnt', 'jb_goods_content_click_click_cnt',
                      'jb_category_content_click_click_cnt', 'jb_search_content_click_click_cnt',
                      'jb_home_page_content_click_click_cnt',
                      'jb_goods_item_click_booster_cnt',
                      'price', 'count_total_sales', 'count_total_browse', 'count_total_favorite', 'count_comment']

    # columns to calculate total_view_count and total_click_count
    VIEW_CNT_COLS = ['jb_merchant_content_impression_impression_cnt', 'jb_goods_content_impression_impression_cnt',
                     'jb_cart_content_impression_impression_cnt', 'jb_category_content_impression_impression_cnt',
                     'jb_search_content_impression_impression_cnt', 'jb_home_page_content_impression_impression_cnt']

    CLICK_CNT_COLS = ['jb_merchant_content_click_click_cnt', 'jb_goods_content_click_click_cnt',
                      'jb_category_content_click_click_cnt', 'jb_search_content_click_click_cnt',
                      'jb_home_page_content_click_click_cnt',
                      'jb_goods_item_click_booster_cnt']

    TOTAL_VIEW_CNT_COL = 'total_view_count'
    TOTAL_CLICK_CNT_COL = 'total_click_count'

    # all categories list file
    COL2CATS_NAMES = 'col2label.pickle'

    # reward normalization steps
    # ex: steps = [1,2,3] -> reward = 3 if reward == 3;
    #                        reward = 2 if (reward < 3 and reward >=2);
    #                        reward = 1 if (reward < 2 and reward >=1)
    REWARD_NORM_STEPS = [1, 2, 3]

    ARM_COL = 'uuid'
    CANDIDATE_COL = 'uuids'
    REWARD_COL = 'reward'

    # below is for postprocess
    DYNAMIC_WEIGHT = 0.01
    RAW_SCORE_COL = 'raw_score'
    NORM_RAW_SCORE_COL = 'norm_raw_score'
    DYNAMIC_SCORE_COL = 'dynamic'
    FINAL_SCORE_COL = 'final_score'
