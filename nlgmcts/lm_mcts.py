import random
from math import *
import math
import numpy as np


class LanguageModelMCTS:
    def __init__(self, language_model, width, text_length, eval_function, c=sqrt(2)):
        self._lm = language_model
        self._width = width
        self._text_length = text_length
        self._eval_function = eval_function
        self._best_sequence = None
        self._c = c

    def search(self, state, num_simulations):
        root_node = _Node(state, self._lm, self._width, self._text_length, self._c)

        # Perform simulations
        for i in range(num_simulations):
            node = root_node

            # Select
            while not node.has_untried_moves() and node.has_children():
                node = node.select_child()

            # Expand
            if node.has_untried_moves():
                move_state = node.select_untried_move()
                node = node.add_child(move_state, self._lm, self._width, self._text_length, self._c)

            # Rollout
            rollout_state = list(node.state)
            while len(rollout_state) < self._text_length:
                rollout_state += [self._select_next_move_randomly(rollout_state, self._lm, self._width)]

            # Backpropagate
            #   backpropagate from the expanded node and work back to the root node
            score = self._eval_function(rollout_state)
            while node is not None:
                node.visits += 1
                node.wins += score
                node = node.parent

            self._store_best(rollout_state, score)

        # return the move that was most visited
        most_visited_node = sorted(root_node.children, key = lambda c: c.visits)[-1]
        return most_visited_node.state

    def _select_next_move_randomly(self, rollout_state, language_model, width):
        top_n_tokens = language_model.top_n_vocab_with_weights(width, ''.join(rollout_state[-self._lm.order() + 1:]))
        return np.random.choice(top_n_tokens[0], p=top_n_tokens[1])

    def _store_best(self, rollout_state, score):
        current_best = self._best_sequence
        if current_best is None or score > current_best[1]:
            self._best_sequence = (rollout_state, score)

    def get_best_sequence(self):
        return self._best_sequence


class _Node:
    def __init__(self, state, language_model, width, text_length, c, parent=None):
        self.state = state
        self._c = c
        self._lm = language_model
        self._width = width
        self._text_length = text_length
        self.wins = 0.0
        self.visits = 0.0
        self.parent = parent
        self.children = []
        self.untried_moves = self._get_child_states()

    def _get_child_states(self):
        top_n_tokens = self._lm.top_n_vocab(self._width, ''.join(self.state[-self._lm.order() + 1:]))
        child_states = []
        for letter in top_n_tokens:
            child_states.append(self.state + [letter])
        return child_states

    def _average_value(self):
        return self.wins / self.visits

    def has_untried_moves(self):
        return self.untried_moves != []

    def select_untried_move(self):
        return random.choice(self.untried_moves)

    def add_child(self, child_state, language_model, width, text_length, c):
        child = _Node(child_state, language_model, width, text_length, c, parent=self)
        self.children.append(child)
        self.untried_moves.remove(child_state)
        return child

    def has_children(self):
        return self.children != []

    def select_child(self):
        highest_ucb1 = None
        selected_child_node = None
        for child_node in self.children:
            ucb1 = child_node.ucb1()
            if highest_ucb1 is None or highest_ucb1 < ucb1:
                highest_ucb1 = ucb1
                selected_child_node = child_node
        return selected_child_node

    def ucb1(self):
        if self.visits == 0.0:
            return math.inf
        return self._average_value() + self._c*sqrt(log(self.parent.visits)/self.visits)
