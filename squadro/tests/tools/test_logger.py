from tempfile import NamedTemporaryFile
from unittest import TestCase

from squadro.tools.disk import load_txt
from squadro.tools.logs import logger


class TestLogger(TestCase):
    def setUp(self):
        ...

    def test_dump_history(self):
        logger.setup(loglevel='INFO')
        logger.debug('debug')
        logger.info('info')
        logger.warn('warn')

        path = NamedTemporaryFile(suffix='.txt').name
        logger.dump_history(path=path)

        text = load_txt(path)
        expected = 'info\nwarn\n'
        self.assertEqual(expected, text)

        logger.clear_history()
        self.assertEqual([], logger.history)

        logger.info('info_again')
        logger.dump_history(path=path)

        text = load_txt(path)
        expected = 'info\nwarn\ninfo_again\n'
        self.assertEqual(expected, text)
