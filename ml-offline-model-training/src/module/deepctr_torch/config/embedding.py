
class FeatureColumnsConfig():
    feature_embedding_size = {
        'gender': 2,
        'site_name': 2,
        'cat1': 16,
        # 'openid': 16,
    }
    dense_feature = ['age', 'semantics_score', 'category_pref_score', 'tag_pref_score']
    behavior_embedding_size = {
        'hist_cat1': 16
    }
