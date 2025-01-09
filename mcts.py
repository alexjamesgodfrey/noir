# mcts.py
import math
import numpy as np
from typing import Dict, List, Tuple
from game import GameState
import torch
from network import DistrictNoirNN


class MCTSNode:
    def __init__(self, game_state: GameState, parent=None, prior_p=1.0):
        self.game_state = game_state
        self.parent = parent
        self.prior_p = prior_p
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.legal_actions = game_state.get_legal_actions()

    def expand(self, policy: np.ndarray):
        for action, prob in enumerate(policy):
            if action in self.legal_actions:
                new_state = self.game_state.clone()
                new_state.make_move(action)
                self.children[action] = MCTSNode(new_state, self, prob)

    def select_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        best_score = float("-inf")
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            # UCB score = Q + U
            # Q = mean value
            # U = exploration bonus
            q_value = -child.value_sum / (child.visit_count + 1)
            u_value = (
                c_puct
                * child.prior_p
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent


class MCTS:
    def __init__(
        self,
        network: DistrictNoirNN,
        num_simulations: int = 800,
        max_depth: int = 20,
        c_puct: float = 1.0,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct

    def search(self, game_state: GameState) -> np.ndarray:
        # If game is terminal or no legal actions, return uniform over legal actions
        if game_state.is_terminal() or not game_state.get_legal_actions():
            legal_actions = game_state.get_legal_actions()
            probs = np.zeros(7)
            if legal_actions:  # If there are legal actions
                probs[legal_actions] = 1.0 / len(legal_actions)
            else:  # If no legal actions, return uniform over all actions
                probs = np.ones(7) / 7
            return probs

        root = MCTSNode(game_state)

        # Initial expansion of root node
        state_tensor = game_state.get_state_tensor().unsqueeze(0)
        with torch.no_grad():
            value, policy = self.network(state_tensor)
        policy = policy.squeeze().numpy()
        root.expand(policy)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection
            depth = 0
            while (
                node.children
                and not node.game_state.is_terminal()
                and depth < self.max_depth
            ):
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
                depth += 1

            # Expansion and evaluation
            if depth >= self.max_depth:
                # If we hit max depth, just evaluate current state
                state_tensor = node.game_state.get_state_tensor().unsqueeze(0)
                with torch.no_grad():
                    value, _ = self.network(state_tensor)
                value = value.item()
            else:
                if not node.game_state.is_terminal():
                    # Normal expansion
                    state_tensor = node.game_state.get_state_tensor().unsqueeze(0)
                    with torch.no_grad():
                        value, policy = self.network(state_tensor)
                    value = value.item()
                    policy = policy.squeeze().numpy()
                    node.expand(policy)
                else:
                    # Terminal state
                    value = node.game_state.get_result()

            # Backup
            for node in search_path:
                node.backup(value)
                value = -value

        # Calculate action probabilities
        visit_counts = np.zeros(7)  # 5 cards + collect + pass
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        # Handle the case where no visits occurred
        if visit_counts.sum() == 0:
            legal_actions = game_state.get_legal_actions()
            probs = np.zeros(7)  # Fixed to 7
            probs[legal_actions] = 1.0 / len(legal_actions)
            return probs

        # Return normalized probabilities
        return visit_counts / visit_counts.sum()
