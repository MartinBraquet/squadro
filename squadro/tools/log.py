import logging


class logger:  # noqa
    client = None
    section = 'main'
    ENABLED_SECTIONS = {
        'main': True,
        'game': True,
        'monte_carlo': True,
        'alpha_beta': True,
        'training': True,
    }

    @classmethod
    def setup(
        cls,
        name: str = None,
        loglevel: str = 'INFO',
        section: str | list = None,
    ):
        """
        Sets up the logger.

        :param name: Name of the logger
        :param loglevel: log level (default: INFO)
        :param section: sections of the logger to render (default: all)
        """
        if cls.client is not None and cls.client.level == logging.INFO:
            return
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
        cls.set_section(section)

    @classmethod
    def stop(cls):
        cls.client = None

    @classmethod
    def set_section(cls, section: str | list) -> None:
        """
        Sets the sections of the logger.

        :param section: Section of the logger to render (default: all)
        """
        if not section:
            return
        if isinstance(section, str):
            section = [section]
        for k, v in cls.ENABLED_SECTIONS.items():
            cls.ENABLED_SECTIONS[k] = k in section

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


class training_logger(logger):  # noqa
    section = 'training'
