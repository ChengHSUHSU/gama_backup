VISUALIZE_CONFIG = {
    'boxplot': {
        'cols': []
    },
    'distribution_histogram': {
        'cols': [],
        'orientation': 'vertical'
    },
    'continuous_distribution_histogram': {
        'cols': [],
        'bins': 100
    },
    'feature_correlation': {
        'cols': [],
    },
    'feature_importance': {
        'cols': [],
        'y_col': 'y'
    }
}


# Example for planet_news_din
PLANET_NEWS_DIN_VIS_CONF = {
    'boxplot': {
        'cols': ['cat1', 'seq_length']
    },
    'distribution_histogram': {
        'cols': ['cat1', 'seq_length'],
        'orientation': 'vertical'
    },
    'continuous_distribution_histogram': {
        'cols': ['category_pref_score', 'user_tag_editor_others',
                 'user_tag_editor_person', 'user_tag_editor_event', 'user_tag_editor_organization',
                 'user_tag_editor_location'],
        'bins': 100
    },
    'feature_correlation': {
        'cols': ['category_pref_score', 'user_tag_editor_others',
                        'user_tag_editor_person', 'user_tag_editor_event', 'user_tag_editor_organization',
                        'user_tag_editor_location'],
    },
    'feature_importance': {
        'cols': ['cat1', 'seq_length', 'category_pref_score', 'user_tag_editor_others',
                        'user_tag_editor_person', 'user_tag_editor_event', 'user_tag_editor_organization',
                        'user_tag_editor_location'],
        'y_col': 'y'
    }
}


FIGSIZE = (10, 5)

