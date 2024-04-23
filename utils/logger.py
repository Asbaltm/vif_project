import logging


class LocalLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalLogger, cls).__new__(cls, *args, **kwargs)
            cls._instance._set_logger()
        return cls._instance

    def _set_logger(self):
        self.logger = logging.getLogger('LocalLogger')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

