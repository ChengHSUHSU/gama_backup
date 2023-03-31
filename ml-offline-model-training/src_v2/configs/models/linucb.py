from src_v2.configs import Config


BaseLinUCBConfigs = Config({
    'ARM_COL': 'content_id',
    'CANDIDATE_COL': 'candidate_neg',
    'REWARD_COL': 'reward'
})
