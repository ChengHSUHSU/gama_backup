
from src_v2.configs import GeneralConfigs
from src_v2.configs.preprocess import BaseHotPreprocessConfigs
from src_v2.configs.models.linucb import BaseLinUCBConfigs

# General Config
PlanetNewsHotGeneralConfig = GeneralConfigs.update({
    'REQUISITE_COLS': ['content_id','content_id_neg', 'reward', 'planet_home_page_impression', 'planet_content_impression', 'home_page_impression'], #DONE
    'CONTENT_TYPE': 'planet_news',
    'SERVICE_TYPE': 'hot',
    'ALERT_MAIL_RECIPIENTS': 'alberthsu@gamania.com'
})

# Data Preprocess Config
PlanetNewsHotLinUCBPreprocessConfig = BaseHotPreprocessConfigs.update({
    # format: {old_name: new_name}
    'COLUMN_TO_RENAME': {},

    # support min-max or z-score (Need to be optimized. Currently normalize method is too hacky)
    'NORMALIZE_COLS': {
        'min-max': {},  # set val for upper bound normalize, set val None to standard min-max normalize
        'z-score': {}
    },
    # chain_configs
    # holds the pipeline of preprocessor for anything you want it to.
    # CHAIN_CONFIGS format: {'func_name': {'func_param_key': 'func_param_value'}}
    'CHAIN_CONFIGS': {'_process': {}}
})

PlanetNewsLinUCBConfig = BaseLinUCBConfigs.update({
    'ARM_COL': 'content_id',
    'CANDIDATE_COL': 'content_id_neg',
    'REWARD_COL': 'reward'
})

PlanetNewsHotLinUCBConfig = PlanetNewsHotGeneralConfig.update(PlanetNewsHotLinUCBPreprocessConfig).update(PlanetNewsLinUCBConfig)
