class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, logger_name='', dev_mode=False, debug_mode=False):
        self._is_dev_mode = dev_mode
        self._is_debug_mode = debug_mode
        self._logger_name = logger_name

    def info(self, *args):
        msg = ' '.join(str(e) for e in args)
        if self._is_dev_mode:
            print('[INFO] ' + msg)
        # self._logger.info(msg)

    def debug(self, *args):
        msg = ' '.join(str(e) for e in args)
        if self._is_debug_mode:
            print('[DEBUG] ' + msg)

    def error(self, *args):
        msg = ' '.join(str(e) for e in args)
        if self._is_dev_mode:
            print('[ERROR] ' + msg)
        # self._logger.error(msg)


logger = Logger(dev_mode=True, debug_mode=False)
