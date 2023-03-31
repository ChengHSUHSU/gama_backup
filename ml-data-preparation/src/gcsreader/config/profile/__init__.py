from src.gcsreader.config.base import Config

BaseProfileConfig = Config({
    'USER_PROFILE_CONDITIONS': {},

    # meta_profile create different fields according to different requirements
    # this would only be needed if gamania user meta profile is used
    'META_COLS': [],

    'USER_PROFILE_JOIN_DAY_DIFF': 5        # TODO: set default join date range to 1 after bug fix
})
