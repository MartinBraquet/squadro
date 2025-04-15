import logging


def setup_logger(name=None, log_file=None, level=logging.INFO):
    formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        # '%(asctime)s %(levelname)s %(message)s',
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
