import torch
from game import GameState, Card, CardType
from network import DistrictNoirNN
from mcts import MCTS


def parse_card_input(input_str: str) -> Card:
    """Convert user input to Card object"""
    card_map = {
        "5": CardType.BLUE_5,
        "6": CardType.PINK_6,
        "7": CardType.ORANGE_7,
        "8": CardType.YELLOW_8,
        "-1": CardType.MOD_NEG_1,
        "-2": CardType.MOD_NEG_2,
        "-3": CardType.MOD_NEG_3,
        "+2": CardType.MOD_POS_2,
        "+3": CardType.MOD_POS_3,
        "+4": CardType.MOD_POS_4,
        "c": CardType.CITY,
        "city": CardType.CITY,
    }

    input_str = input_str.lower().strip()
    if input_str in card_map:
        return Card(card_map[input_str])
    raise ValueError(f"Invalid card input: {input_str}")


def get_card_list_from_user(prompt: str) -> list[Card]:
    """Get a list of cards from user input"""
    print(prompt)
    print("Enter cards one per line. Valid inputs: 5,6,7,8,-1,-2,-3,+2,+3,+4,c/city")
    print("Enter a blank line when done")

    cards = []
    while True:
        try:
            card_input = input("> ").strip()
            if not card_input:
                break
            cards.append(parse_card_input(card_input))
        except ValueError as e:
            print(f"Error: {e}")
    return cards


def advisor_loop(model_path: str, num_simulations: int = 800):
    round_real = 1

    """Main advisor loop"""
    # Initialize AI
    network = DistrictNoirNN()
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # If you saved the entire checkpoint, you'll have something like:
    #   {
    #       'model_state_dict': ...,
    #       'optimizer_state_dict': ...,
    #       'iteration': ...,
    #       'params': ...,
    #       'training_history': ...,
    #       ...
    #   }

    # 1. Reconstruct your DistrictNoirNN either with or without extra params (if required):
    network = DistrictNoirNN()  # or DistrictNoirNN(**checkpoint["params"]) if needed

    # 2. Load the actual model weights from checkpoint['model_state_dict']:
    # network.load_state_dict(checkpoint["model_state_dict"])
    model_state_dict = checkpoint["model_state_dict"]
    filtered_state_dict = {
        k: v
        for k, v in model_state_dict.items()
        if k in network.state_dict() and network.state_dict()[k].shape == v.shape
    }
    network.load_state_dict(filtered_state_dict, strict=False)

    network.eval()
    mcts = MCTS(network, num_simulations=num_simulations)

    # Initialize game state
    game_state = GameState()

    # ***** IMPORTANT: We do NOT track opponent's hand in advisor mode *****
    game_state.player2_hand = []

    # Display initial info
    print("\nDistrict Noir AI Advisor")
    print("========================")
    print(f"Random role assignment:")
    print(f"Player {game_state.paperboy} is Paperboy")
    print(f"Player {game_state.spy} is Spy")
    print(f"Player {game_state.first_player} goes first")

    # Ask user about initial middle cards & your hand
    game_state.middle_cards = get_card_list_from_user(
        "\nWhat cards are in the middle to start?"
    )
    game_state.player1_hand = get_card_list_from_user("\nWhat cards are in your hand?")

    while not game_state.is_terminal():
        print("\nCurrent Game State:")
        print("===================")
        print(f"Round {game_state.round_number}")

        print("\nMiddle cards (oldest → newest):")
        if not game_state.middle_cards:
            print("(empty)")
        else:
            for i, card in enumerate(game_state.middle_cards):
                if i == len(game_state.middle_cards) - 1:
                    print(f"{card.type.name} ← Most recent")
                else:
                    print(f"{card.type.name} → ", end="")

        # We do not show player2_hand, because it is unknown
        print("\n\nOpponent's hand: (Unknown)")
        print("\nYour hand:")
        print([card.type.name for card in game_state.player1_hand])

        print("\nYour collected cards:")
        print([card.type.name for card in game_state.player1_collected])

        print("\nOpponent's collected cards (public info):")
        print([card.type.name for card in game_state.player2_collected])

        if game_state.current_player == 1:  # Your turn
            mcts_probs = mcts.search(game_state)
            legal_actions = game_state.get_legal_actions()

            print("\nRecommended actions (in order of preference):")
            actions_with_probs = [(act, mcts_probs[act]) for act in legal_actions]
            actions_with_probs.sort(key=lambda x: x[1], reverse=True)

            for action, prob in actions_with_probs:
                if action == 5:
                    print(
                        f"Collect the last {min(5, len(game_state.middle_cards))} cards (confidence: {prob:.2%})"
                    )
                elif action == 6:
                    print(f"Pass (confidence: {prob:.2%})")
                else:
                    card_play = game_state.player1_hand[action]
                    print(f"Play {card_play.type.name} (confidence: {prob:.2%})")

            # Ask user what they actually did
            while True:
                if not game_state.player1_hand:
                    # No cards to play
                    if (
                        game_state.can_player1_collect
                        and len(game_state.middle_cards) > 0
                    ):
                        action_type = "c"
                        print("Auto-collecting (no cards in hand).")
                    else:
                        action_type = "pass"
                        print("Auto-passing (no cards in hand and cannot collect).")
                else:
                    action_type = input(
                        "\nWhat did you do? (p=play card, c=collect): "
                    ).lower()

                # Play a card
                if action_type == "p" and game_state.player1_hand:
                    print("Which card did you play?")
                    played_card = parse_card_input(input("> "))
                    # Remove that card from your hand
                    for i, c in enumerate(game_state.player1_hand):
                        if c.type == played_card.type:
                            game_state.player1_hand.pop(i)
                            break
                    # Add to middle
                    game_state.middle_cards.append(played_card)
                    break

                # Collect
                elif (
                    action_type == "c"
                    and game_state.can_player1_collect
                    and len(game_state.middle_cards) > 0
                ):
                    cards_to_collect = min(5, len(game_state.middle_cards))
                    collected = game_state.middle_cards[-cards_to_collect:]
                    game_state.player1_collected.extend(collected)
                    del game_state.middle_cards[-cards_to_collect:]
                    game_state.can_player1_collect = False
                    print(f"You collected {len(collected)} cards.")
                    break

                # Pass
                elif action_type == "pass" and not game_state.player1_hand:
                    print("You passed.")
                    break

            # End-of-round check
            if (
                not game_state.player1_hand
                and game_state.player2_cards_played_this_round >= 5
                and not game_state.can_player1_collect
                and not game_state.can_player2_collect
            ):
                print("\nRound is over!")
                round_real += 1
                game_state.round_number = round_real
                print(f"\nStarting Round {game_state.round_number}")
                next_first = (
                    game_state.first_player
                    if game_state.round_number % 2 == 1
                    else 3 - game_state.first_player
                )
                print(f"Player {next_first} goes first")

                # Ask for new hand
                game_state.player1_hand = get_card_list_from_user(
                    "What cards did you draw?"
                )
                # Reset collecting & card-play count
                game_state.can_player1_collect = True
                game_state.can_player2_collect = True
                game_state.player2_cards_played_this_round = 0
                game_state.current_player = next_first
            else:
                # Switch to opponent
                game_state.current_player = 2

        else:  # Opponent's turn
            print("\nWhat did your opponent do?")
            # If we assume no knowledge of their hand
            # we rely purely on user input to track if they 'played' or 'collected'
            if game_state.player2_cards_played_this_round < 5:
                action_type = input("(p=play card, c=collect, pass=pass): ").lower()
            else:
                # If they've played 5 times, they either collect or pass
                if game_state.can_player2_collect and len(game_state.middle_cards) > 0:
                    print(
                        "Opponent must collect, since they've played 5 cards already."
                    )
                    action_type = "c"
                else:
                    print("Opponent must pass, since they've played 5 cards already.")
                    action_type = "pass"

            if action_type == "p" and game_state.player2_cards_played_this_round < 5:
                print(
                    "Which card did they play? (You won't remove it from their hand, just track it)"
                )
                played_card = parse_card_input(input("> "))
                # We do NOT remove from player2_hand because it's unknown
                game_state.middle_cards.append(played_card)
                game_state.player2_cards_played_this_round += 1
            elif action_type == "c" and game_state.can_player2_collect:
                cards_to_collect = min(5, len(game_state.middle_cards))
                collected_cards = game_state.middle_cards[-cards_to_collect:]
                game_state.player2_collected.extend(collected_cards)
                game_state.middle_cards = game_state.middle_cards[:-cards_to_collect]
                game_state.can_player2_collect = False
                print(f"Opponent collected {len(collected_cards)} cards.")
            else:
                print("Opponent passed.")

            # End-of-round check
            if (
                not game_state.player1_hand
                and game_state.player2_cards_played_this_round >= 5
                and not game_state.can_player1_collect
                and not game_state.can_player2_collect
            ):
                print("\nRound is over!")
                round_real += 1
                game_state.round_number = round_real
                print(f"\nStarting Round {game_state.round_number}")
                next_first = (
                    game_state.first_player
                    if game_state.round_number % 2 == 1
                    else 3 - game_state.first_player
                )
                print(f"Player {next_first} goes first")

                # Ask for new hand
                game_state.player1_hand = get_card_list_from_user(
                    "What cards did you draw?"
                )
                # Reset collecting & card-play count
                game_state.can_player1_collect = True
                game_state.can_player2_collect = True
                game_state.player2_cards_played_this_round = 0
                game_state.current_player = next_first
            else:
                game_state.current_player = 1

    # Game Over
    print("\nGame Over!")
    p1_score = game_state._calculate_score(game_state.player1_collected)
    p2_score = game_state._calculate_score(game_state.player2_collected)
    p1_cities = sum(
        1 for card in game_state.player1_collected if card.type == CardType.CITY
    )
    p2_cities = sum(
        1 for card in game_state.player2_collected if card.type == CardType.CITY
    )

    if p1_cities == 3:
        print("You win by collecting all city cards!")
    elif p2_cities == 3:
        print("Opponent wins by collecting all city cards!")
    else:
        print(f"Final scores – You: {p1_score}, Opponent: {p2_score}")
        if p1_score > p2_score:
            print("You win on points!")
        elif p2_score > p1_score:
            print("Opponent wins on points!")
        else:
            print("It's a tie!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="District Noir AI Advisor")
    parser.add_argument(
        "--model",
        type=str,
        default="models/iterations/district_noir_iter004_sims400_games30_depth12_lr0.005000_loss0.9888.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )
    args = parser.parse_args()
    advisor_loop(args.model, args.simulations)
