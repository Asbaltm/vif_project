from utils.logger import LocalLogger


class Base:

    @property
    def logger(self):
        return LocalLogger().logger
