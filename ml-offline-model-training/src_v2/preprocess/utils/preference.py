import pandas as pd
import json
import ast


class Preference():

    def __init__(self, config):
        self.config = config

    def _calculate_user_pref_score(self, user_data, item_data):
        """Dot product calculation of user data and item_data"""
        result = 0.0
        if isinstance(item_data, str):
            result = user_data.get(item_data, 0.0)      # One-hot data
        elif isinstance(item_data, list):
            for val in item_data:
                result += user_data.get(val, 0.0)       # Multi-hot data
        return result

    def _parse_tags_from_string(self, tags):
        """Parse tags from string to list, and extract text value"""
        result = []
        if isinstance(tags, str):
            tags = ast.literal_eval(tags)
        for val in tags:
            if '-' in val and len(val) > 2:
                result.append(val[2:])
            else:
                result.append(val)
        return result

    def _get_pref_scores_mapping(self, x, level=['click', 'cat1']):
        """Parse user category profile from string to dict, and extract key-value pair of preference score"""
        result = {}

        x = json.loads(x).get(level[0], {}) \
                         .get(level[1], {})

        for key, val in x.items():
            result[self._remove_entity_prefix(key)] = val.get('pref', 0.0)
        return result

    def _remove_entity_prefix(self, text):
        return text[text.find('-') + 1:]

    def extract_category_pref_score(self, df, level=['click', 'cat1'], cat_col='user_category', score_col='category_pref_score', enable_single_user=False):
        # Preprocess of raw category profile parsing

        if enable_single_user:
            """
            enable_single_user is used for single user ranking, performance increase: O(N^2) -> O(N)
            Model training: `enable_single_user=False`
            Online serving: `enable_single_user=True`
            """
            user_cat_dict = self._get_pref_scores_mapping(df[cat_col].iloc[0], level=level)
            df[score_col] = df.apply(lambda x: self._calculate_user_pref_score(user_cat_dict, x[level[1]]), axis=1)
        else:
            df['parsed_user_cat'] = df[cat_col].apply(lambda x: self._get_pref_scores_mapping(x, level=level) if pd.notna(x) else {})
            df[score_col] = df.apply(lambda x: self._calculate_user_pref_score(x['parsed_user_cat'], x[level[1]]), axis=1)

        return df

    def extract_tag_pref_score(self, df, tag_entity_list=[], user_tag_col='user_tag', item_tag_col='tags', tagging_type='editor', score_col='', enable_single_user=False):

        for tag_key in tag_entity_list:
            tagging_column_name = score_col if score_col else f'user_tag_{tagging_type}_{tag_key}'

            if enable_single_user:
                """
                enable_single_user is used for single user ranking, performance increase: O(N^2) -> O(N)
                Model training: `enable_single_user=False`
                Online serving: `enable_single_user=True`
                """
                user_tagging_dict = self._get_pref_scores_mapping(df[user_tag_col].iloc[0], level=[tagging_type, tagging_column_name])
                df[tagging_column_name] = df.apply(lambda x: self._calculate_user_pref_score(user_tagging_dict, x[item_tag_col]), axis=1)
            else:
                df[tagging_column_name] = df[user_tag_col].apply(lambda x: self._get_pref_scores_mapping(x, level=[
                                                                 tagging_type, tag_key]) if pd.notna(x) else {})
                df[tagging_column_name] = df.apply(lambda x: self._calculate_user_pref_score(x[tagging_column_name], x[item_tag_col]), axis=1)
        return df
