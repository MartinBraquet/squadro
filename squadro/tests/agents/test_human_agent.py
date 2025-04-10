from unittest import TestCase
from unittest.mock import patch

import pygame

from squadro.agents.human_agent import HumanAgent
from squadro.squadro_state import SquadroState


class MockedEvent:
    def __init__(self, click_type):
        self.type = click_type


class TestGame(TestCase):
    def setUp(self):
        self.agent = HumanAgent()

    def test_get_name(self):
        self.assertEqual(self.agent.get_name(), 'human')


class TestPyGame(TestCase):
    def setUp(self):
        self.agent = HumanAgent()
        self.state = SquadroState(first=0)
        pygame.init()

    def tearDown(self):
        pygame.quit()

    @patch('pygame.event.get', lambda: [MockedEvent(pygame.MOUSEBUTTONUP)])
    @patch('pygame.mouse.get_pos', lambda: (150, 650))
    def test_get_action(self):
        action = self.agent.get_action(self.state)
        self.assertEqual(action, 0)

    @patch('pygame.event.get', lambda: [MockedEvent(pygame.QUIT)])
    def test_quit(self):
        with self.assertRaises(SystemExit):
            self.agent.get_action(self.state)
