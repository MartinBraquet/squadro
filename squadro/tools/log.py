import logging


class logger:  # noqa
    client = None
    section = 'main'
    ENABLED_SECTIONS = {
        'main': True,
        'game': True,
        'monte_carlo': True,
        'alpha_beta': True,
    }

    @classmethod
    def setup(cls, name: str = None, loglevel: str = 'INFO'):
        cls.client = logging.getLogger(name)
        cls.client.setLevel(loglevel)
        handler = logging.StreamHandler()
        handler.setLevel(loglevel)
        formatter = logging.Formatter(
            # '%(asctime)s - '
            # '%(name)s - '
            # '%(levelname)s - '
            '%(message)s'
        )
        handler.setFormatter(formatter)
        cls.client.addHandler(handler)

    @classmethod
    def stop(cls):
        cls.client = None

    @classmethod
    def log(cls, msg, level=logging.INFO, **kwargs):
        if cls.client is None or not cls.ENABLED_SECTIONS[cls.section]:
            return
        cls.client.log(msg=msg, level=level, stacklevel=3, **kwargs)

    @classmethod
    def debug(cls, msg, **kwargs):
        cls.log(msg=msg, level=logging.DEBUG, **kwargs)

    @classmethod
    def info(cls, msg, **kwargs):
        cls.log(msg=msg, **kwargs)

    @classmethod
    def warn(cls, msg, **kwargs):
        cls.log(msg=msg, level=logging.WARNING, **kwargs)

    @classmethod
    def error(cls, msg, **kwargs):
        cls.log(msg=msg, level=logging.ERROR, **kwargs)

    @classmethod
    def critical(cls, msg, **kwargs):
        cls.log(msg=msg, level=logging.CRITICAL, **kwargs)


class game_logger(logger):  # noqa
    section = 'game'


class monte_carlo_logger(logger):  # noqa
    section = 'monte_carlo'


class alpha_beta_logger(logger):  # noqa
    section = 'alpha_beta'
