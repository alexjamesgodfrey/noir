import os
from game import GameState  # Changed from game to game_state
from network import DistrictNoirNN
from typing import Tuple, List
from mcts import MCTS
import numpy as np
import torch
import torch.nn.functional as F


# training.py
def self_play_game(network: DistrictNoirNN, mcts: MCTS) -> List[Tuple]:
    print("\nStarting new self-play game")
    game_state = GameState()
    training_data = []

    move_count = 0
    while not game_state.is_terminal():
        print(f"\nMove {move_count}")
        move_count += 1
        # Get MCTS probabilities
        mcts_probs = mcts.search(game_state)

        # Ensure probabilities are valid
        if np.any(np.isnan(mcts_probs)):
            print("Warning: NaN probabilities detected")
            legal_actions = game_state.get_legal_actions()
            mcts_probs = np.zeros_like(mcts_probs)
            mcts_probs[legal_actions] = 1.0 / len(legal_actions)

        # Store state and probabilities
        training_data.append(
            (
                game_state.get_state_tensor(),
                torch.tensor(mcts_probs),
                game_state.current_player,
            )
        )

        # Select action (with temperature)
        temperature = 1.0 if len(game_state.deck) > 10 else 0.5
        if temperature == 0:
            action = mcts_probs.argmax()
        else:
            mcts_probs = mcts_probs ** (1 / temperature)
            mcts_probs /= mcts_probs.sum()
            legal_actions = game_state.get_legal_actions()

            # Ensure we only choose from legal actions
            masked_probs = np.zeros_like(mcts_probs)
            masked_probs[legal_actions] = mcts_probs[legal_actions]
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            else:
                masked_probs[legal_actions] = 1.0 / len(legal_actions)

            action = np.random.choice(len(mcts_probs), p=masked_probs)

        # game_state.pretty_print_state()
        print("Player " + str(game_state.current_player) + " makes action:", action)

        # # press any key to continue
        # input()

        # Make move
        game_state.make_move(action)

    # Get game result and add to training data
    result = game_state.get_result()
    return [
        (state, probs, result * (1 if player == game_state.current_player else -1))
        for state, probs, player in training_data
    ]


def train_network(
    network: DistrictNoirNN,
    num_iterations: int = 20,
    games_per_iteration: int = 100,
    num_simulations: int = 200,
    max_depth: int = 20,
):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
    mcts = MCTS(network, num_simulations=num_simulations, max_depth=max_depth)

    # Create directory for model saves if it doesn't exist
    os.makedirs("models", exist_ok=True)

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        training_data = []

        # Self-play phase
        for game_num in range(games_per_iteration):
            print(
                f"Playing game {game_num + 1}/{games_per_iteration} of iteration {iteration + 1}"
            )
            game_data = self_play_game(network, mcts)
            training_data.extend(game_data)

        # Training phase
        total_loss = 0
        num_batches = 0
        for epoch in range(5):
            np.random.shuffle(training_data)
            for batch_idx in range(0, len(training_data), 32):
                batch = training_data[batch_idx : batch_idx + 32]
                states, mcts_probs, results = zip(*batch)

                states = torch.stack(states)
                mcts_probs = torch.stack(mcts_probs)
                results = torch.tensor(results).float().unsqueeze(1)

                optimizer.zero_grad()
                value_pred, policy_pred = network(states)

                value_loss = F.mse_loss(value_pred, results)
                policy_loss = -torch.mean(
                    torch.sum(mcts_probs * torch.log(policy_pred + 1e-8), dim=1)
                )
                loss = value_loss + policy_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Iteration {iteration + 1} complete. Average loss: {avg_loss:.4f}")

        # Save network periodically
        if (iteration + 1) % 10 == 0:
            save_path = f"models/district_noir_iter_{iteration+1}.pth"
            torch.save(network.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Always save latest model
        latest_path = "models/district_noir_latest.pth"
        torch.save(network.state_dict(), latest_path)
        print(f"Latest model saved to {latest_path}")
