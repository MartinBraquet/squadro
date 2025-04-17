import logging
from logging.config import dictConfig

from pydantic import BaseModel


class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "squadro"
    LOG_LEVEL: str = "INFO"

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "fmt": '%(levelprefix)s | %(asctime)s | %(name)s | %(filename)s | %(funcName)s() | L%(lineno)-4d %('
                   'message)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    }
    handlers: dict = dict(
        default={
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    )
    loggers: dict = {
        "squadro": {"handlers": list(handlers.keys()), "level": LOG_LEVEL},
    }


def setup_logger(name: str = 'squadro', loglevel: str = None):
    dictConfig(LogConfig().model_dump())
    logger.client = logging.getLogger(name)
    logger.client.propagate = False
    if loglevel is not None:
        logger.client.setLevel(loglevel)


def stop_logger():
    logger.client = None


class logger:  # noqa
    client = None
    reported = {}

    @classmethod
    def log(cls, msg, level=logging.INFO, **kwargs):
        if cls.client is None:
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


logger_DISABLED = {'main': False, 'memory': False, 'tourney': False, 'mcts': False, 'model': False}
