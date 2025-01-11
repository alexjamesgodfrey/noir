import os
from game import GameState  # Changed from game to game_state
from network import DistrictNoirNN
from typing import Tuple, List
from mcts import MCTS
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# training.py
def self_play_game(network: DistrictNoirNN, mcts: MCTS) -> List[Tuple]:
    game_state = GameState()
    training_data = []
    pbar = tqdm(
        total=48,
        desc="Game Progress",
        bar_format="{l_bar}{bar}| {n_fmt} / max {total_fmt}",
    )
    last_move_count = 0
    move_count = 0

    while not game_state.is_terminal():
        move_count += 1
        # Get MCTS probabilities
        mcts_probs = mcts.search(game_state)

        # Ensure probabilities are valid
        if np.any(np.isnan(mcts_probs)):
            tqdm.write("Warning: NaN probabilities detected")
            legal_actions = game_state.get_legal_actions()
            mcts_probs = np.zeros_like(mcts_probs)
            mcts_probs[legal_actions] = 1.0 / len(legal_actions)

        # Update progress bar by 1 move
        pbar.update(1)
        last_move_count += 1

        # Store state and probabilities
        training_data.append(
            (
                game_state.get_state_tensor(),
                torch.tensor(mcts_probs),
                game_state.current_player,
            )
        )

        # Select action (with temperature)
        deck_size = len(game_state.deck)
        total_cards = 48  # adjust based on your game
        if deck_size > 0.75 * total_cards:  # First quarter
            temperature = 1.2
        elif deck_size > 0.5 * total_cards:  # Second quarter
            temperature = 0.8
        elif deck_size > 0.25 * total_cards:  # Third quarter
            temperature = 0.5
        else:  # Last quarter
            temperature = 0.3
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

        game_state.make_move(action)

    pbar.close()
    # Get game result and add to training data
    result = game_state.get_result()
    tqdm.write(f"Game completed in {last_move_count} moves with result: {result}\n")

    return [
        (state, probs, result * (1 if player == game_state.current_player else -1))
        for state, probs, player in training_data
    ]


import os
import json
import datetime
from game import GameState
from network import DistrictNoirNN
from typing import Tuple, List
from mcts import MCTS
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def format_model_name(iteration, avg_loss, params):
    """Create a detailed model name with parameters and performance metrics."""
    return (
        f"district_noir_"
        f"iter{iteration:03d}_"
        f"sims{params['num_simulations']}_"
        f"games{params['games_per_iteration']}_"
        f"depth{params['max_depth']}_"
        f"lr{params['learning_rate']:.6f}_"
        f"loss{avg_loss:.4f}"
    )


def train_network(
    network: DistrictNoirNN,
    num_iterations: int = 20,
    games_per_iteration: int = 100,
    num_simulations: int = 200,
    max_depth: int = 20,
):
    # Store start time for duration calculation
    training_start_time = datetime.datetime.now()

    # Initialize parameters dictionary
    params = {
        "num_iterations": num_iterations,
        "games_per_iteration": games_per_iteration,
        "num_simulations": num_simulations,
        "max_depth": max_depth,
        "learning_rate": 0.008,
        "batch_size": 64,
        "epochs_per_iter": 10,
        "max_grad_norm": 1.0,
        "value_loss_weight": 0.75,
        "policy_loss_weight": 1.0,
        "initial_lr": 0.05,
        "lr_decay_factor": 0.005,
        "lr_decay_iterations": 10,
        "patience": 10,
        "min_delta": 0.05,
    }

    # Create directories for different types of model saves
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/iterations", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=params["learning_rate"])
    mcts = MCTS(network, num_simulations=num_simulations, max_depth=max_depth)

    best_loss = float("inf")
    training_history = []

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
        for epoch in range(params["epochs_per_iter"]):
            np.random.shuffle(training_data)
            for batch_idx in range(0, len(training_data), params["batch_size"]):
                batch = training_data[batch_idx : batch_idx + params["batch_size"]]
                states, mcts_probs, results = zip(*batch)

                states = torch.stack(states)
                mcts_probs = torch.stack(mcts_probs)
                results = torch.tensor(results).float().unsqueeze(1)

                optimizer.zero_grad()
                value_pred, policy_pred = network(states)

                value_loss = params["value_loss_weight"] * F.mse_loss(
                    value_pred, results
                )
                policy_loss = params["policy_loss_weight"] * -torch.mean(
                    torch.sum(mcts_probs * torch.log(policy_pred + 1e-8), dim=1)
                )
                loss = value_loss + policy_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), params["max_grad_norm"]
                )
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Iteration {iteration + 1} complete. Average loss: {avg_loss:.4f}")

        # Store training metrics
        training_history.append(
            {
                "iteration": iteration + 1,
                "avg_loss": avg_loss,
                "games_played": games_per_iteration,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
            }
        )

        # Generate model name with parameters
        model_name = format_model_name(iteration + 1, avg_loss, params)

        # Save iteration checkpoint
        iter_path = f"models/iterations/{model_name}.pth"
        torch.save(
            {
                "iteration": iteration + 1,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "params": params,
                "training_history": training_history,
            },
            iter_path,
        )

        # Save best model if current loss is better
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"models/best/{model_name}_best.pth"
            torch.save(
                {
                    "iteration": iteration + 1,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "params": params,
                    "training_history": training_history,
                },
                best_path,
            )
            print(f"New best model saved with loss: {avg_loss:.4f}")

        # Add after training metrics storage in the iteration loop
        if (iteration + 1) % params["lr_decay_iterations"] == 0:
            # Decay learning rate
            new_lr = optimizer.param_groups[0]["lr"] * params["lr_decay_factor"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate decayed to: {new_lr:.6f}")

        # Save periodic checkpoints (every 10 iterations)
        if (iteration + 1) % 10 == 0:
            checkpoint_path = f"models/checkpoints/{model_name}_checkpoint.pth"
            torch.save(
                {
                    "iteration": iteration + 1,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "params": params,
                    "training_history": training_history,
                },
                checkpoint_path,
            )

    # Save final training summary
    training_summary = {
        "training_history": training_history,
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "training_params": params,
        "total_games": num_iterations * games_per_iteration,
        "training_duration": str(datetime.datetime.now() - training_start_time),
        "final_value_loss": value_loss.item(),
        "final_policy_loss": policy_loss.item(),
    }

    summary_filename = f"models/training_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_filename, "w") as f:
        json.dump(training_summary, f, indent=4)

    print(f"\nTraining completed!")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Training summary saved to: {summary_filename}")

    return training_history
