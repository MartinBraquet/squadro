from squadro import Game
from squadro.agents.montecarlo_agent import MonteCarloQLearningAgent
from squadro.evaluators.evaluator import QLearningEvaluator
from squadro.state import State
from squadro.tools.constants import DefaultParams, DATA_PATH
from squadro.tools.log import training_logger as logger


class QLearningTrainer:
    def __init__(
        self,
        n_pawns=None,
        lr=None,
        gamma=None,
        n_steps=None,
        eval_interval=None,
        eval_steps=None,
        model_path=None
    ):
        """
        :param n_pawns: number of pawns in the game.
        :param lr: learning rate.
        :param gamma: discount factor.
        :param n_steps: number of training steps.
        :param eval_interval: interval at which to evaluate the agent.
        :param eval_steps: number of steps to evaluate the agent.
        :param model_path: path to save the model.
        """
        self.n_pawns = n_pawns or DefaultParams.n_pawns
        self.lr = lr or .2
        self.gamma = gamma or .95
        self.n_steps = n_steps or int(5e4)
        self.eval_interval = eval_interval or int(1e3)
        self.eval_steps = eval_steps or int(100)

        self.agent = 'mcts_q_learning'
        model_path = model_path or DATA_PATH / f"q_table_{self.n_pawns}.json"
        self.evaluator = QLearningEvaluator(model_path=model_path)
        self.evaluator_old = QLearningEvaluator(model_path=model_path.replace(".json", "_old.json"))

    def run(self) -> None:
        """
        Run the training loop.
        """
        self.evaluator.dump(self.evaluator_old.model_path)

        # Should be close to 50% as the agents are the same
        logger.info(self.evaluate_agent())

        for n in range(self.n_steps):
            game = Game(
                n_pawns=self.n_pawns,
                agent_0=self.agent,
                agent_1=self.agent,
                save_states=True
            )
            game.run()

            self.back_propagate(game.state_history.copy())

            if (n + 1) % self.eval_interval == 0:
                ev_random = self.evaluate_agent(vs='random')
                ev = self.evaluate_agent(vs='initial')
                logger.info(
                    f"{n + 1} steps"
                    f", {ev * 100:.2f}% vs initial"
                    f", {ev_random * 100:.2f}% vs random"
                )
                self.evaluator.dump()
                if ev > .9:
                    self.evaluator.dump(self.evaluator_old.model_path)

    def back_propagate(self, state_history: list[State]) -> None:
        """
        Update the Q-table based on the reward obtained at the end of the game.

        :param state_history: List of states visited during the game.
        :return: None
        """
        state = state_history.pop()
        reward = self.evaluator.get_value(state)
        assert reward in {-1, 1}
        value = self.Q[self.evaluator.get_id(state)] = reward

        while state_history:
            value = - value
            state = state_history.pop()
            state_id = self.evaluator.get_id(state)
            value = (1 - self.lr) * self.Q.get(state_id, 0) + self.lr * self.gamma * value
            self.Q[state_id] = value

    @property
    def Q(self) -> dict:  # noqa
        return self.evaluator.Q

    def evaluate_agent(self, vs='initial') -> float:
        """
        Evaluate the success rate of the current agent against another agent.
        """
        if vs == 'initial':
            vs = MonteCarloQLearningAgent(evaluator=self.evaluator_old)

        v = 0
        for n in range(self.eval_steps):
            g = Game(
                n_pawns=self.n_pawns,
                agent_0=self.agent,
                agent_1=vs,
            )
            g.run()
            v += g.winner

        return v / self.eval_steps
