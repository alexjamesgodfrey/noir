# play.py
import torch
from network import DistrictNoirNN
from game import GameState
from mcts import MCTS
import argparse


def display_card(card):
    """Format card name for display"""
    return card.type.name


def display_cards(cards):
    """Display list of cards with indices"""
    return " ".join(f"{i}:{display_card(card)}" for i, card in enumerate(cards))


def display_game_state(game_state: GameState):
    """Display current game state in a readable format"""
    print("\n" + "=" * 50)
    print(f"\nMiddle cards ({len(game_state.middle_cards)} cards):")
    if game_state.middle_cards:
        print(display_cards(game_state.middle_cards))
    else:
        print("(empty)")

    print(f"\nYour hand ({len(game_state.player1_hand)} cards):")
    if game_state.player1_hand:
        print(display_cards(game_state.player1_hand))
    else:
        print("(empty)")

    print("\nCollected cards:")
    print(f"You: {len(game_state.player1_collected)} cards")
    print(
        f"   City cards: {sum(1 for card in game_state.player1_collected if card.type.name == 'CITY')}"
    )
    print(
        f"   Sets completed: {game_state._count_complete_sets(game_state.player1_collected)}"
    )

    print(f"\nAI: {len(game_state.player2_collected)} cards")
    print(
        f"   City cards: {sum(1 for card in game_state.player2_collected if card.type.name == 'CITY')}"
    )
    print(
        f"   Sets completed: {game_state._count_complete_sets(game_state.player2_collected)}"
    )

    print("\nCurrent scores:")
    p1_score = game_state._calculate_score(game_state.player1_collected)
    p2_score = game_state._calculate_score(game_state.player2_collected)
    print(f"You: {p1_score}")
    print(f"AI: {p2_score}")
    print("=" * 50 + "\n")


def get_human_move(game_state: GameState) -> int:
    """Get and validate human player's move"""
    legal_actions = game_state.get_legal_actions()

    while True:
        print("\nAvailable actions:")
        if len(game_state.player1_hand) > 0:
            print("0-4: Play card from hand")
        if 5 in legal_actions:
            print("5: Collect cards")

        try:
            action = int(input("\nEnter your move: "))
            if action in legal_actions:
                return action
            else:
                print("Invalid move! Try again.")
        except ValueError:
            print("Please enter a number!")


def play_game(model_path: str, num_simulations: int = 800):
    """Play a game against the AI"""
    # Load the model
    network = DistrictNoirNN(input_size=197)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    # Initialize MCTS
    mcts = MCTS(network, num_simulations=num_simulations)

    # Initialize game
    game_state = GameState()

    print("\nWelcome to District Noir!")
    print("You are Player 1, AI is Player 2")

    while not game_state.is_terminal():
        display_game_state(game_state)

        if game_state.current_player == 1:
            # Human turn
            print("\nYour turn!")
            action = get_human_move(game_state)
        else:
            # AI turn
            print("\nAI is thinking...")
            mcts_probs = mcts.search(game_state)
            action = mcts_probs.argmax()
            if action == 5:
                print("AI chooses to collect cards")
            else:
                print(f"AI plays card {action}")

        game_state.make_move(action)

    # Game over
    display_game_state(game_state)

    # Show final result
    p1_score = game_state._calculate_score(game_state.player1_collected)
    p2_score = game_state._calculate_score(game_state.player2_collected)
    p1_cities = sum(
        1 for card in game_state.player1_collected if card.type.name == "CITY"
    )
    p2_cities = sum(
        1 for card in game_state.player2_collected if card.type.name == "CITY"
    )

    print("\nGame Over!")
    if p1_cities == 3:
        print("You win by collecting all city cards!")
    elif p2_cities == 3:
        print("AI wins by collecting all city cards!")
    else:
        print(f"Final scores - You: {p1_score}, AI: {p2_score}")
        if p1_score > p2_score:
            print("You win by points!")
        elif p2_score > p1_score:
            print("AI wins by points!")
        else:
            print("It's a tie!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play District Noir against AI")
    parser.add_argument(
        "--model",
        type=str,
        default="models/district_noir_latest.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )

    args = parser.parse_args()
    play_game(args.model, args.simulations)
