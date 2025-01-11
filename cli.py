from game import GameState
from network import DistrictNoirNN
import torch
from mcts import MCTS
from training import train_network


# cli.py
def display_game_state(game_state: GameState):
    print("\nMiddle cards:")
    for i, card in enumerate(game_state.middle_cards):
        print(f"{i}: {card.type.name}", end=" ")
    print("\n")

    print(f"Player {game_state.current_player}'s hand:")
    for i, card in enumerate(game_state.get_current_hand()):
        print(f"{i}: {card.type.name}", end=" ")
    print("\n")

    print("Collected cards:")
    print("Player 1:", [card.type.name for card in game_state.player1_collected])
    print("Player 2:", [card.type.name for card in game_state.player2_collected])
    print()


def human_vs_ai():
    network = DistrictNoirNN(input_size=52)
    network.load_state_dict(torch.load("district_noir_latest.pth"))
    network.eval()
    mcts = MCTS(network)

    game_state = GameState()
    human_player = 1  # Human is player 1

    while not game_state.is_terminal():
        # display_game_state(game_state)

        if game_state.current_player == human_player:
            # Human turn
            legal_actions = game_state.get_legal_actions()
            while True:
                print("Actions: 0-4 to play card, 5 to collect")
                action = int(input("Enter your action: "))
                if action in legal_actions:
                    break
                print("Invalid action, try again")
        else:
            # AI turn
            mcts_probs = mcts.search(game_state)
            action = mcts_probs.argmax()
            print(f"AI chooses action: {action}")

        game_state.make_move(action)

    # Game over
    result = game_state.get_result()
    if result > 0:
        print("Player 1 wins!")
    elif result < 0:
        print("Player 2 wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()

    if args.train:
        network = DistrictNoirNN(input_size=52)
        train_network(
            network,
            num_iterations=10,
            games_per_iteration=25,
            num_simulations=600,
            max_depth=12,
        )
    elif args.play:
        human_vs_ai()
    else:
        print("Please specify --train or --play")
