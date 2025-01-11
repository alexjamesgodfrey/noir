import os
import json
import datetime
import multiprocessing
from functools import partial
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from game import GameState  # or from game_state import GameState
from network import DistrictNoirNN
from mcts import MCTS


####################################
# 1) HELPER: Random Opponent Logic #
####################################
def random_policy_action(game_state: GameState):
    """Return a random valid action from game_state."""
    legal_actions = game_state.get_legal_actions()
    return np.random.choice(legal_actions)


def play_game_vs_random(
    network: DistrictNoirNN, num_simulations: int = 100, max_depth: int = 10
) -> int:
    """
    Plays one full game between:
      - MCTS-based policy (using `network`)
      - random policy (pure random valid moves).
    Returns +1 if the network's side wins, -1 if the random side wins, 0 if tie.

    By convention, let's say `network` is always 'player 1',
    and 'player 2' is random. Adjust as needed if your game engine has
    symmetrical or forced player indexing.
    """
    game_state = GameState()
    # Force the game state to track that "current_player == 1" is the network
    # and "current_player == 2" is random. If your GameState logic differs,
    # adapt accordingly.

    # MCTS for the network
    mcts = MCTS(network, num_simulations=num_simulations, max_depth=max_depth)

    while not game_state.is_terminal():
        current_player = game_state.current_player

        if current_player == 1:
            # network's turn
            mcts_probs = mcts.search(game_state)
            # handle any NaNs:
            if np.any(np.isnan(mcts_probs)):
                legal_actions = game_state.get_legal_actions()
                mcts_probs = np.zeros_like(mcts_probs)
                mcts_probs[legal_actions] = 1.0 / len(legal_actions)

            # pick action greedily or with some small temperature
            action = np.argmax(mcts_probs)
            # or do a temperature-based sampling, your choice
        else:
            # random's turn
            action = random_policy_action(game_state)

        game_state.make_move(action)

    # Evaluate outcome from the perspective of 'player 1' (network)
    result = game_state.get_result()
    # Usually result is +1 if the last player to move won, -1 if lost, 0 if tie, etc.
    # But if your game logic is different, adapt accordingly.

    # If your GameState doesn't store who specifically won,
    # you might do something like:
    # winner = game_state.get_winner()  # returns 1 for net, 2 for random, etc.
    # return +1 if winner == 1 else -1 if winner == 2 else 0

    return result


#####################################################
# 2) PARALLEL SELF-PLAY (for collecting training data)
#####################################################
def self_play_single_game(
    network: DistrictNoirNN, num_simulations: int, max_depth: int
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Runs a single self-play game with the given network and MCTS parameters.
    Returns a list of (state, policy, final_result_for_that_player).
    """
    game_state = GameState()
    mcts = MCTS(network, num_simulations=num_simulations, max_depth=max_depth)

    training_data = []
    pbar = tqdm(
        total=48,
        desc="Game Progress",
        bar_format="{l_bar}{bar}| {n_fmt} / max {total_fmt}",
    )
    last_move_count = 0

    while not game_state.is_terminal():
        mcts_probs = mcts.search(game_state)
        if np.any(np.isnan(mcts_probs)):
            tqdm.write("Warning: NaN probabilities detected")
            legal_actions = game_state.get_legal_actions()
            mcts_probs = np.zeros_like(mcts_probs)
            mcts_probs[legal_actions] = 1.0 / len(legal_actions)

        pbar.update(1)
        last_move_count += 1

        # Store (state, policy, current_player)
        training_data.append(
            (
                game_state.get_state_tensor(),
                torch.tensor(mcts_probs),
                game_state.current_player,
            )
        )

        # Temperature logic
        deck_size = len(game_state.deck)
        total_cards = 48
        if deck_size > 0.75 * total_cards:
            temperature = 1.2
        elif deck_size > 0.5 * total_cards:
            temperature = 0.8
        elif deck_size > 0.25 * total_cards:
            temperature = 0.5
        else:
            temperature = 0.3

        if temperature == 0:
            action = mcts_probs.argmax()
        else:
            mcts_probs = mcts_probs ** (1 / temperature)
            mcts_probs /= mcts_probs.sum()
            legal_actions = game_state.get_legal_actions()
            masked_probs = np.zeros_like(mcts_probs)
            masked_probs[legal_actions] = mcts_probs[legal_actions]
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            else:
                masked_probs[legal_actions] = 1.0 / len(legal_actions)
            action = np.random.choice(len(mcts_probs), p=masked_probs)

        game_state.make_move(action)

    pbar.close()
    result = game_state.get_result()
    tqdm.write(f"Game completed in {last_move_count} moves with result: {result}\n")

    # Convert each record to (state, policy, final_value_for_that_player)
    final_data = []
    for state_tensor, policy_tensor, player in training_data:
        value_for_player = result if player == game_state.current_player else -result
        final_data.append((state_tensor, policy_tensor, value_for_player))

    return final_data


def self_play_worker(
    idx: int,
    network_state_dict: dict,
    num_simulations: int,
    max_depth: int,
    device: str = "cpu",
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    local_network = DistrictNoirNN().to(device)
    local_network.load_state_dict(network_state_dict)
    local_network.eval()

    return self_play_single_game(local_network, num_simulations, max_depth)


def parallel_self_play(
    network: DistrictNoirNN,
    num_games: int,
    num_simulations: int,
    max_depth: int,
    device: str = "cpu",
    num_workers: int = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    network_state_dict = network.state_dict()
    worker_func = partial(
        self_play_worker,
        network_state_dict=network_state_dict,
        num_simulations=num_simulations,
        max_depth=max_depth,
        device=device,
    )

    print(f"Starting parallel self-play: {num_games} games, {num_workers} workers.")
    all_data = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iter = pool.imap_unordered(worker_func, range(num_games))
        for game_result in tqdm(results_iter, total=num_games, desc="Self-play"):
            all_data.extend(game_result)

    return all_data


##############################
# 3) FORMAT MODEL NAME UTILITY
##############################
def format_model_name(iteration, avg_loss, params):
    return (
        f"district_noir_"
        f"iter{iteration:03d}_"
        f"sims{params['num_simulations']}_"
        f"games{params['games_per_iteration']}_"
        f"depth{params['max_depth']}_"
        f"lr{params['learning_rate']:.6f}_"
        f"loss{avg_loss:.4f}"
    )


#########################
# 4) MAIN TRAINING ROUTINE
#########################
def train_network(
    network: DistrictNoirNN,
    num_iterations: int = 10,
    games_per_iteration: int = 25,
    num_simulations: int = 100,
    max_depth: int = 10,
):
    training_start_time = datetime.datetime.now()

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
        "lr_decay_factor": 0.003,
        "lr_decay_iterations": 10,
        "patience": 10,
        "min_delta": 0.05,
    }

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/iterations", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=params["learning_rate"])
    best_loss = float("inf")
    training_history = []

    for iteration in range(num_iterations):
        print(f"\n======== Iteration {iteration + 1}/{num_iterations} ========")

        # ---------------------------
        # A) COLLECT TRAINING DATA
        # ---------------------------
        training_data = parallel_self_play(
            network,
            num_games=games_per_iteration,
            num_simulations=num_simulations,
            max_depth=max_depth,
            device="cpu",
            num_workers=None,  # or set to e.g. 4
        )

        # -----------------------
        # B) TRAIN THE NETWORK
        # -----------------------
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

        avg_loss = total_loss / num_batches if num_batches else 0.0
        print(f"Iteration {iteration + 1} complete. Average loss: {avg_loss:.4f}")

        # -----------------------------
        # C) EVALUATE VS RANDOM POLICY
        # -----------------------------
        print("Evaluating vs random policy (10 games)...")
        network_wins = 0
        random_wins = 0
        draws = 0
        num_eval_games = 10
        for _ in range(num_eval_games):
            result = play_game_vs_random(
                network, num_simulations=num_simulations, max_depth=max_depth
            )
            # result: +1 => network wins, -1 => random wins, 0 => tie
            if result > 0:
                network_wins += 1
            elif result < 0:
                random_wins += 1
            else:
                draws += 1

        print(
            f"Network vs Random - Wins: {network_wins}, Losses: {random_wins}, Draws: {draws}"
        )
        evaluation_stats = {
            "network_wins": network_wins,
            "random_wins": random_wins,
            "draws": draws,
        }

        # ---------------------
        # D) SAVE / LOG RESULTS
        # ---------------------
        # Store training stats
        training_history.append(
            {
                "iteration": iteration + 1,
                "avg_loss": avg_loss,
                "games_played": games_per_iteration,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
                "evaluation_vs_random": evaluation_stats,
            }
        )

        # Save iteration checkpoint
        model_name = format_model_name(iteration + 1, avg_loss, params)
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

        # Best model
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

        # LR decay every 'lr_decay_iterations'
        if (iteration + 1) % params["lr_decay_iterations"] == 0:
            new_lr = optimizer.param_groups[0]["lr"] * params["lr_decay_factor"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate decayed to: {new_lr:.6f}")

        # Periodic checkpoint
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

    # -------------------------------------
    # Final summary
    # -------------------------------------
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


###########################################
# 5) OPTIONAL: Example usage if run directly
###########################################
if __name__ == "__main__":
    # Initialize a fresh network
    model = DistrictNoirNN()

    # Train for 10 iterations, 25 self-play games each iteration,
    # and do quick 100-simulation MCTS
    train_network(
        model,
        num_iterations=50,
        games_per_iteration=25,
        num_simulations=400,
        max_depth=30,
    )
