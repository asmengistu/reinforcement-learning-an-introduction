#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import pickle

BOARD_SIZE = 3
# The board is represented by an array of size n^2,
DATA_SIZE = BOARD_SIZE**2


class CELL:
    EMPTY = 0
    X = 1
    O = 2


class State(object):
    # Build list of row/column/diagonal indexers used for determing winner.
    _rows, _cols = [], []
    for i in range(BOARD_SIZE):
        _rows.append(range(i * BOARD_SIZE, (i + 1) * BOARD_SIZE))
        # For example range(1, 9, 3) == [1, 4, 5]
        _cols.append(range(i, DATA_SIZE, BOARD_SIZE))
    _diags = [
        range(0, DATA_SIZE, BOARD_SIZE + 1),  # e.g. 0, 4, 8
        range(BOARD_SIZE - 1, DATA_SIZE - 1, BOARD_SIZE - 1),  # e.g. 2, 4, 6
    ]
    _indexers = _rows + _cols + _diags

    def __init__(self, data=np.zeros(DATA_SIZE), is_xs_turn=True):
        self.data = data
        self.winner = None
        self._is_terminal = None
        self.is_xs_turn = is_xs_turn

    def __hash__(self):
        return hash(tuple(self.data))

    def is_terminal(self):
        """Whether the state is terminal."""
        if self._is_terminal is not None:
            return self._is_terminal
        for indexer in State._indexers:
            # Get values in indexer (either row/col/diag)
            values = self.data[indexer]
            if np.all(values == CELL.X):
                self.winner = CELL.X
                self._is_terminal = True
                return True
            if np.all(values == CELL.O):
                self.winner = CELL.O
                self._is_terminal = True
                return True

        # Check for tie. Tie happens when there are no empty squares left and
        # there is no winner.
        if np.sum(self.data == CELL.EMPTY) == 0:
            self.winner = CELL.EMPTY
            self._is_terminal = True
            return True

        self._is_terminal = False
        return False

    def next_state(self, location):
        """Returns a new state with the next player having marked `location`.

        First player is assumed to be `CELL.X`.
        """
        assert not self.is_terminal(), 'Attempted to move at terminal state!'
        new_state = State(np.copy(self.data), not self.is_xs_turn)
        new_state.data[location] = CELL.X if self.is_xs_turn else CELL.O
        return new_state

    def _get_char(self, location):
        if self.data[location] == CELL.EMPTY:
            return '-'
        if self.data[location] == CELL.X:
            return 'X'
        if self.data[location] == CELL.O:
            return 'O'

    # print the board
    def show(self):
        for i in range(BOARD_SIZE):
            print(' -------------')
            chars = [''] + map(self._get_char, State._rows[i]) + ['']
            print(' | '.join(chars))
        print(' -------------')


def get_all_states_impl(current_state, all_states):
    for i in range(0, DATA_SIZE):
        if current_state.data[i] == CELL.EMPTY:
            new_state = current_state.next_state(i)
            if hash(new_state) not in all_states.keys():
                is_terminal = new_state.is_terminal()
                all_states[hash(new_state)] = new_state
                if not is_terminal:
                    get_all_states_impl(new_state, all_states)


def get_all_states():
    curr_state = State()
    all_states = dict()
    all_states[hash(curr_state)] = curr_state
    get_all_states_impl(curr_state, all_states)
    return all_states

# all possible board configurations
all_states = get_all_states()


class Judger:
    """Runs a single game between agents until terminal state is reached.

    Keyword arguments:
    player1 -- first player. starts (X).
    player2 -- second player. (O)
    feedback -- whether players receive rewards at termination. (default True)
    """
    def __init__(self, player1, player2, feedback=True):
        self.p1 = player1
        self.p2 = player2
        self._feedback = feedback
        self.p1_symbol = CELL.X
        self.p2_symbol = CELL.O
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def give_reward(self):
        if self.current_state.winner == self.p1_symbol:
            self.p1.feed_reward(1)
            self.p2.feed_reward(0)
        elif self.current_state.winner == self.p2_symbol:
            self.p1.feed_reward(0)
            self.p2.feed_reward(1)
        else:
            self.p1.feed_reward(0)
            self.p2.feed_reward(0)

    def feed_current_state(self):
        self.p1.feed_state(self.current_state)
        self.p2.feed_state(self.current_state)

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.current_state = State()

    # @show: if True, print each terminal board during the game
    def play(self, show=False):
        self.reset()
        self.feed_current_state()
        while True:
            curr_player = self.p1 if self.current_state.is_xs_turn else self.p2
            if show:
                self.current_state.show()
            location = curr_player.take_action()
            self.current_state = self.current_state.next_state(location)
            self.feed_current_state()
            if self.current_state.is_terminal():
                if self._feedback:
                    self.give_reward()
                return self.current_state.winner


class BasePlayer(object):
    def __init__(self):
        super(BasePlayer, self).__init__()
        self.current_state = None

    def reset(self):
        pass

    def set_symbol(self, symbol):
        pass

    def feed_state(self, state):
        pass

    def feed_reward(self, reward):
        pass

    def take_action(self):
        raise NotImplementedError()

    def save_policy(self):
        pass

    def load_policy(self):
        pass


# Temporal-Difference learning AI agent with reward.
class RewardTdPlayer(BasePlayer):
    # @step_size: step size to update estimations
    # @explore_rate: possibility to explore (epsilon in epsilon-greedy)
    def __init__(self, step_size=0.1, explore_rate=0.1):
        super(RewardTdPlayer, self).__init__()
        self.estimations = dict()
        self.step_size = step_size
        self.explore_rate = explore_rate
        self.states = []

    def reset(self):
        self.states = []

    def set_symbol(self, symbol):
        self.symbol = symbol
        for state_id in all_states.keys():
            state = all_states[state_id]
            if state.is_terminal():
                if state.winner == self.symbol:
                    self.estimations[state_id] = 1.0
                else:
                    self.estimations[state_id] = 0
            else:
                self.estimations[state_id] = 0.5

    # accept a state
    def feed_state(self, state):
        self.states.append(state)

    # update estimation according to reward
    def feed_reward(self, reward):
        if len(self.states) == 0:
            return
        self.states = [hash(state) for state in self.states]
        target = reward
        for latest_state in reversed(self.states):
            latest_est = self.estimations[latest_state]
            value = latest_est + self.step_size * (target - latest_est)
            self.estimations[latest_state] = value
            target = value
        self.states = []

    # determine next action
    def take_action(self):
        state = self.states[-1]
        possible_actions = np.where(state.data == CELL.EMPTY)[0]
        possible_states = map(hash, map(state.next_state, possible_actions))

        # Maybe explore
        if np.random.binomial(1, self.explore_rate):
            # Not sure if truncating is the best way to deal with exploratory step
            # Maybe it's better to only skip this step rather than forget all the history
            self.states = []
            action = np.random.choice(possible_actions)
            return action

        values = []
        for state_id, action in zip(possible_states, possible_actions):
            values.append((self.estimations[state_id], action))
        values.sort(key=lambda x: x[0], reverse=True)
        # print(len(values))
        # print(values[0])
        action = values[0][1]
        return action

    def save_policy(self):
        with open('/tmp/optimal_policy_' + str(self.symbol), 'wb') as fw:
            pickle.dump(self.estimations, fw)

    def load_policy(self):
        with open('/tmp/optimal_policy_' + str(self.symbol), 'rb') as fr:
            self.estimations = pickle.load(fr)


# human interface
# input a number to put a chessman
# | 1 | 2 | 3 |
# | 4 | 5 | 6 |
# | 7 | 8 | 9 |
class HumanPlayer(BasePlayer):
    def __init__(self):
        super(HumanPlayer, self).__init__()
        self.symbol = None
        self.current_state = None
        return

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def feed_state(self, state):
        self.current_state = state
        return

    def take_action(self):
        response = int(input("Input your position (1-9):"))
        response -= 1
        if self.current_state.data[response] != 0:
            return self.take_action()
        return response


class RandomPlayer(BasePlayer):
    def __init__(self):
        super(RandomPlayer, self).__init__()
        self.current_state = None

    def feed_state(self, state):
        self.current_state = state

    def take_action(self):
        possible_actions = np.where(self.current_state.data == CELL.EMPTY)[0]
        return np.random.choice(possible_actions)


def train(epochs=20000):
    print('Starting training...')
    player1 = RewardTdPlayer()
    player2 = RewardTdPlayer()
    judger = Judger(player1, player2)
    p1_wins = 0.0
    p2_wins = 0.0
    for i in range(1, epochs+1):
        winner = judger.play()
        if winner == CELL.X:
            p1_wins += 1
        if winner == CELL.O:
            p2_wins += 1
        if i % 100 == 0:
            print("Epoch %d: (%f, %f)" % (i, p1_wins / i, p2_wins / i))
        judger.reset()
    player1.save_policy()
    player2.save_policy()


def compete(turns=500,
            player1=RewardTdPlayer(explore_rate=0),
            player2=RewardTdPlayer(explore_rate=0)):
    print('Self-play evaluation')
    judger = Judger(player1, player2, False)
    player1.load_policy()
    player2.load_policy()
    p1_wins = 0.0
    p2_wins = 0.0
    for i in range(1, turns+1):
        winner = judger.play()
        if winner == CELL.X:
            p1_wins += 1
        if winner == CELL.O:
            p2_wins += 1
        if i % 100 == 0:
            print("Epoch %d: (%f, %f)" % (i, p1_wins / i, p2_wins / i))
        judger.reset()
    print(p1_wins / turns)
    print(p2_wins / turns)


def play(agent=RewardTdPlayer(explore_rate=0)):
    while True:
        player1 = agent
        player2 = HumanPlayer()
        judger = Judger(player1, player2, False)
        player1.load_policy()
        winner = judger.play(True)
        if winner == player2.symbol:
            print("Win!")
        elif winner == player1.symbol:
            print("Lose!")
        else:
            print("Tie!")

if __name__ == '__main__':
    train()
    compete(player2=RandomPlayer(), player1=RewardTdPlayer(explore_rate=0))
    # play()
