import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CardType(Enum):
    # Number cards
    BLUE_5 = auto()
    PINK_6 = auto()
    ORANGE_7 = auto()
    YELLOW_8 = auto()

    # Modifier cards
    MOD_NEG_1 = auto()
    MOD_NEG_2 = auto()
    MOD_NEG_3 = auto()
    MOD_POS_2 = auto()
    MOD_POS_3 = auto()
    MOD_POS_4 = auto()

    # Special cards
    CITY = auto()


@dataclass
class Card:
    type: CardType

    def get_one_hot(self) -> np.ndarray:
        """Convert card to one-hot encoding"""
        encoding = np.zeros(len(CardType))
        encoding[self.type.value - 1] = 1  # -1 because enum starts at 1
        return encoding

    @staticmethod
    def get_card_index(card_type: CardType) -> int:
        """Get the index in the one-hot encoding for a card type"""
        return card_type.value - 1


class GameState:
    def __init__(self):
        self.paperboy = None  # 1 or 2
        self.spy = None  # 1 or 2
        self.first_player = None  # 1 or 2
        self.current_player = None  # 1 or 2

        # Initialize card counts
        self.card_counts = {
            CardType.BLUE_5: 5,
            CardType.PINK_6: 6,
            CardType.ORANGE_7: 7,
            CardType.YELLOW_8: 8,
            CardType.MOD_NEG_1: 3,
            CardType.MOD_NEG_2: 4,
            CardType.MOD_NEG_3: 2,
            CardType.MOD_POS_2: 4,
            CardType.MOD_POS_3: 2,
            CardType.MOD_POS_4: 1,
            CardType.CITY: 3,
        }

        # Initialize game components
        self.deck = self._initialize_deck()
        self.player1_hand: List[Card] = []
        self.player2_hand: List[Card] = []
        self.player2_cards_played_this_round = 0
        self.middle_cards: List[Card] = []
        self.player1_collected: List[Card] = []
        self.player2_collected: List[Card] = []
        self.removed_cards: List[Card] = []

        # Game state flags
        self.current_player = 1  # 1 or 2
        self.can_player1_collect = True
        self.can_player2_collect = True
        self.round_number = 1  # 1 to 4

        # Initialize game
        self._setup_game()

    def _initialize_deck(self) -> List[Card]:
        """Create and shuffle the initial deck"""
        deck = []
        for card_type, count in self.card_counts.items():
            deck.extend([Card(card_type) for _ in range(count)])
        np.random.shuffle(deck)
        return deck

    def _setup_game(self):
        """Perform initial game setup"""
        self.paperboy = np.random.choice([1, 2])
        self.spy = 3 - self.paperboy

        # Randomly choose whether paperboy or spy goes first
        self.first_player = np.random.choice([1, 2])
        self.current_player = self.first_player

        # Remove 3 random cards
        self.removed_cards = self.deck[:3]
        self.deck = self.deck[3:]

        # Deal initial hands
        self.player1_hand = self.deck[:5]
        self.deck = self.deck[5:]
        self.player2_hand = self.deck[:5]
        self.deck = self.deck[5:]

        # Place initial middle cards
        self.middle_cards = self.deck[:2]
        self.deck = self.deck[2:]

    def get_state_tensor(self) -> torch.Tensor:
        """
        Convert current game state to tensor representation with imperfect information,
        always from Player 1's perspective.

        This version expands the feature set to encode more strategic details.
        """

        features = []

        # -------------------------------------------------------------------------
        # 1. Card location features (same logic, but keep if you find it useful)
        # -------------------------------------------------------------------------
        # We currently track the total counts of each CardType in:
        #   - Your hand
        #   - Middle cards
        #   - Your collected
        #   - Opponent collected
        # (Removed cards are excluded in this example, but you can add them back.)
        card_location_features = []
        for cards in [
            self.player1_hand,  # Always your hand
            self.middle_cards,
            self.player1_collected,
            self.player2_collected,
        ]:
            encoding = np.zeros(len(CardType))
            for card in cards:
                encoding[Card.get_card_index(card.type)] += 1
            card_location_features.extend(encoding)
        features.extend(card_location_features)
        # If you want these to remain 44 features (4 * 11), that’s fine.
        # Right now, that’s 4 groups × 11 = 44 (instead of 55, since we commented out removed cards).

        # -------------------------------------------------------------------------
        # 2. Basic game state features
        # -------------------------------------------------------------------------
        game_state_features = []

        # Who is current player? (1 if it's you, else 0)
        game_state_features.append(float(self.current_player == 1))

        # Whether each player can still collect
        game_state_features.append(float(self.can_player1_collect))
        game_state_features.append(float(self.can_player2_collect))

        # Round encoding: If your game absolutely never exceeds round 4, you can keep 4 slots,
        # else you can expand to 5 or more:
        round_encoding = np.zeros(4)
        # Safely clamp the round_number in case it goes out of range:
        round_index = min(self.round_number - 1, 3)  # clamp at 3
        round_encoding[round_index] = 1
        game_state_features.extend(round_encoding)

        # Example: You could also add "role" encoding:
        #   - role_encoding = np.zeros(2)
        #   - if self.paperboy == 1: role_encoding[0] = 1  # player1 is paperboy
        #   - else: role_encoding[1] = 1  # player1 is spy
        # game_state_features.extend(role_encoding)

        features.extend(game_state_features)

        # -------------------------------------------------------------------------
        # 3. Derived features
        # -------------------------------------------------------------------------
        derived_features = []

        # 3.1 - More granular set completion features
        derived_features.extend(self._get_expanded_set_completion_features())

        # 3.2 - Color majority features (you can keep the old [-1..1] approach
        #       AND also add raw difference approach, or total fraction).
        derived_features.extend(self._get_color_majority_features())
        derived_features.extend(self._get_color_majority_diff_features())  # new

        # 3.3 - Special card features (number of sets completed, city card difference, etc.)
        derived_features.extend(self._get_special_card_features())

        # 3.4 - Probability of each card type still unaccounted for
        derived_features.extend(self._get_probability_features())

        # 3.5 - Position embeddings for the last N cards in the middle
        derived_features.extend(self._get_position_embeddings())

        # 3.6 - Historical moves (placeholder or actual)
        derived_features.extend(self._get_historical_moves())

        # 3.7 - Extended winning-condition features
        derived_features.extend(self._get_winning_condition_features())

        # 3.8 - Actual/estimated score difference
        derived_features.extend(self._get_score_difference_features())

        # 3.9 - Hand composition details (example: how many city cards or modifiers in your hand)
        derived_features.extend(self._get_hand_composition_features())

        # 3.10 - Opponent collect threat or forced move
        derived_features.extend(self._get_opponent_collect_threat_features())

        # Add all derived features to main feature list
        features.extend(derived_features)

        # -------------------------------------------------------------------------
        # Return as Torch tensor
        # -------------------------------------------------------------------------
        return torch.tensor(features, dtype=torch.float32)

    def _get_expanded_set_completion_features(self) -> List[float]:
        """
        Returns multiple features that gauge progress toward set completion,
        not just for (5-6-7-8) but also partial sets, etc.
        """
        features = []

        # Current perspective's relevant cards:
        if self.current_player == 1:
            current_hand = self.player1_hand
            current_collected = self.player1_collected
        else:
            current_hand = self.player2_hand
            current_collected = self.player2_collected

        # 1) Main set completion ratio (already in original code)
        main_set = [
            CardType.BLUE_5,
            CardType.PINK_6,
            CardType.ORANGE_7,
            CardType.YELLOW_8,
        ]
        total_main_set_cards = sum(
            1 for c in current_hand + current_collected if c.type in main_set
        )
        completion = total_main_set_cards / len(main_set)  # range 0..1
        features.append(completion)

        # 2) How many total main sets have you completed?
        completed_sets = self._count_complete_sets(current_collected)
        features.append(float(completed_sets))

        # 3) Maybe how many of each rank in your hand+collected (like partial synergy):
        #    This is another vector of length 4 or so:
        for rank_type in main_set:
            count = sum(
                1 for c in current_hand + current_collected if c.type == rank_type
            )
            features.append(float(count))

        # You can add more expansions if you’d like.
        # e.g. track if you have 2-of-a-kind, 3-of-a-kind, etc.

        return features

    def _get_color_majority_diff_features(self) -> List[float]:
        """
        Calculate absolute difference in number of each color (P1 minus P2 or P2 minus P1),
        regardless of who is current player. This can help the network see raw advantage.
        """
        colors = [
            CardType.BLUE_5,
            CardType.PINK_6,
            CardType.ORANGE_7,
            CardType.YELLOW_8,
        ]
        features = []

        p1_counts = {c: 0 for c in colors}
        p2_counts = {c: 0 for c in colors}

        for card in self.player1_collected:
            if card.type in p1_counts:
                p1_counts[card.type] += 1
        for card in self.player2_collected:
            if card.type in p2_counts:
                p2_counts[card.type] += 1

        for color in colors:
            diff = p1_counts[color] - p2_counts[color]
            features.append(float(diff))  # could be negative or positive

        return features

    def _get_score_difference_features(self) -> List[float]:
        """
        Returns a single feature: (your_score - opponent_score).
        If it's your turn (P1), we do p1_score - p2_score.
        Else we do p2_score - p1_score, to keep the perspective consistent.
        """
        p1_score = self._calculate_score(self.player1_collected)
        p2_score = self._calculate_score(self.player2_collected)

        if self.current_player == 1:
            diff = p1_score - p2_score
        else:
            diff = p2_score - p1_score

        return [float(diff)]

    def _get_hand_composition_features(self) -> List[float]:
        """
        Returns counts of special categories in your hand (if it's your turn).
        For opponent's turn, you might encode "unknown" or skip.
        """
        features = []

        current_hand = (
            self.player1_hand if self.current_player == 1 else self.player2_hand
        )

        # Count city cards in hand
        city_in_hand = sum(1 for c in current_hand if c.type == CardType.CITY)
        features.append(float(city_in_hand))

        # Count positive modifiers in hand
        pos_mods = [CardType.MOD_POS_2, CardType.MOD_POS_3, CardType.MOD_POS_4]
        pmods_in_hand = sum(1 for c in current_hand if c.type in pos_mods)
        features.append(float(pmods_in_hand))

        # Count negative modifiers in hand
        neg_mods = [CardType.MOD_NEG_1, CardType.MOD_NEG_2, CardType.MOD_NEG_3]
        nmods_in_hand = sum(1 for c in current_hand if c.type in neg_mods)
        features.append(float(nmods_in_hand))

        return features

    def _get_opponent_collect_threat_features(self) -> List[float]:
        """
        Example: 2 features
        1. Whether the opponent can still collect in this round
        2. Whether the opponent is 'forced' to collect (e.g., they have no cards)
        """
        features = []

        # Opponent ID:
        opp = 2 if self.current_player == 1 else 1

        can_opp_collect = (
            self.can_player2_collect if opp == 2 else self.can_player1_collect
        )
        features.append(float(can_opp_collect))

        # If the opponent is forced to collect (no cards in hand but can still collect)
        # This is a simple example:
        opp_hand_size = len(self.player2_hand) if opp == 2 else len(self.player1_hand)
        forced = opp_hand_size == 0 and can_opp_collect and len(self.middle_cards) > 0
        features.append(float(forced))

        return features

    def _get_set_completion_features(self) -> List[float]:
        """Calculate set completion progress (0-1) for each possible set"""
        current_player_cards = self.player1_hand
        current_player_collected = (
            self.player1_collected
            if self.current_player == 1
            else self.player2_collected
        )

        # We need to return exactly 4 features - one for each possible set
        features = []

        # Main set (5-6-7-8)
        main_set = [
            CardType.BLUE_5,
            CardType.PINK_6,
            CardType.ORANGE_7,
            CardType.YELLOW_8,
        ]
        collected_count = sum(
            1 for card in current_player_collected if card.type in main_set
        )
        hand_count = sum(1 for card in current_player_cards if card.type in main_set)
        completion = (collected_count + hand_count) / len(main_set)
        features.append(completion)

        # Add three more potential set types (for future expansion)
        features.extend([0.0, 0.0, 0.0])  # Placeholder for other set types

        return features  # Should return exactly 4 features

    def _get_color_majority_features(self) -> List[float]:
        """Calculate majority status (-1 to 1) for each color"""
        colors = [
            CardType.BLUE_5,
            CardType.PINK_6,
            CardType.ORANGE_7,
            CardType.YELLOW_8,
        ]
        features = []

        for color in colors:
            p1_count = sum(1 for card in self.player1_collected if card.type == color)
            p2_count = sum(1 for card in self.player2_collected if card.type == color)
            total_possible = self.card_counts[color]

            # Scale to [-1, 1] based on majority status
            if self.current_player == 1:
                majority = (p1_count - p2_count) / total_possible
            else:
                majority = (p2_count - p1_count) / total_possible
            features.append(majority)

        return features

    def _get_special_card_features(self) -> List[float]:
        """Calculate special card related features"""
        features = []

        # Complete set count
        p1_sets = self._count_complete_sets(self.player1_collected)
        p2_sets = self._count_complete_sets(self.player2_collected)
        features.append(
            p1_sets - p2_sets if self.current_player == 1 else p2_sets - p1_sets
        )

        # Modifier card differential
        p1_mods = sum(
            1
            for card in self.player1_collected
            if card.type in [CardType.MOD_POS_2, CardType.MOD_POS_3, CardType.MOD_POS_4]
        )
        p2_mods = sum(
            1
            for card in self.player2_collected
            if card.type in [CardType.MOD_POS_2, CardType.MOD_POS_3, CardType.MOD_POS_4]
        )
        features.append(
            p1_mods - p2_mods if self.current_player == 1 else p2_mods - p1_mods
        )

        # City card count
        p1_cities = sum(
            1 for card in self.player1_collected if card.type == CardType.CITY
        )
        p2_cities = sum(
            1 for card in self.player2_collected if card.type == CardType.CITY
        )
        features.append(
            p1_cities - p2_cities if self.current_player == 1 else p2_cities - p1_cities
        )

        return features

    def _get_probability_features(self) -> List[float]:
        """Calculate probability of each card type being available"""
        features = []

        visible_cards = (
            self.player1_hand
            + self.middle_cards
            + self.player1_collected
            + self.player2_collected
        )

        for card_type in CardType:
            total = self.card_counts[card_type]
            visible = sum(1 for card in visible_cards if card.type == card_type)
            prob = max(0, total - visible - len(self.removed_cards)) / total
            features.append(prob)

        return features

    def _get_position_embeddings(self) -> List[float]:
        """Create position-aware embeddings for middle cards"""
        features = []

        # Create 11-dimensional embedding for each of the last 5 positions
        for i in range(min(5, len(self.middle_cards))):
            card = self.middle_cards[-(i + 1)]  # Get cards from newest to oldest
            embedding = card.get_one_hot()
            features.extend(embedding)

        # Pad with zeros if needed
        padding_needed = 5 - len(self.middle_cards)
        if padding_needed > 0:
            features.extend([0] * (padding_needed * len(CardType)))

        return features

    def _get_historical_moves(self) -> List[float]:
        """Encode the last 5 moves made in the game"""
        # In practice, you would maintain a move history in the GameState
        # For now, return placeholder zeros
        return [0] * (5 * len(CardType))

    def _get_winning_condition_features(self) -> List[float]:
        """Calculate distance to winning conditions"""
        features = []

        # Cities needed for city victory
        current_cities = (
            sum(1 for card in self.player1_collected if card.type == CardType.CITY)
            if self.current_player == 1
            else sum(1 for card in self.player2_collected if card.type == CardType.CITY)
        )
        features.append((3 - current_cities) / 3)  # Normalize to [0, 1]

        # Approximate point differential needed for point victory
        # This would require a more complex calculation in practice
        features.append(0.5)  # Placeholder

        return features

    def _count_complete_sets(self, cards: List[Card]) -> int:
        """Count number of complete sets (5-6-7-8) in a collection of cards"""
        counts = {
            CardType.BLUE_5: 0,
            CardType.PINK_6: 0,
            CardType.ORANGE_7: 0,
            CardType.YELLOW_8: 0,
        }

        for card in cards:
            if card.type in counts:
                counts[card.type] += 1

        return min(counts.values())

    def display_game_state(self):
        print("\nMiddle cards:")
        for i, card in enumerate(self.middle_cards):
            print(f"{i}: {card.type.name}", end=" ")
        print("\n")

        print(f"Player {self.current_player}'s hand:")
        for i, card in enumerate(self.get_current_hand()):
            print(f"{i}: {card.type.name}", end=" ")
        print("\n")

        print("Collected cards:")
        print("Player 1:", [card.type.name for card in self.player1_collected])
        print("Player 2:", [card.type.name for card in self.player2_collected])
        print()

    def get_legal_actions(self) -> List[int]:
        """Get list of legal actions (0-4 for playing cards, 5 for collect, 6 for pass)"""
        actions = []
        current_hand = self.get_current_hand()

        # Can play any card from hand
        if current_hand:
            actions.extend(range(len(current_hand)))

        # Can collect anytime if haven't collected yet and there are cards
        can_collect = (
            (self.current_player == 1 and self.can_player1_collect)
            or (self.current_player == 2 and self.can_player2_collect)
        ) and len(self.middle_cards) > 0

        # print("can collect:", can_collect)

        if can_collect:
            actions.append(5)

        # Can pass only if no cards in hand AND either:
        # 1. Can't collect (already collected) OR
        # 2. No cards to collect
        has_cards_in_hand = (
            len(self.player1_hand) > 0
            if self.current_player == 1
            else self.player2_cards_played_this_round < 5
        )
        # print("len(self.player1_hand):", len(self.player1_hand))
        # print("current player:", self.current_player)
        # print(
        #     "self.player2_cards_played_this_round:",
        #     self.player2_cards_played_this_round,
        # )
        # print("has_cards_in_hand:", has_cards_in_hand)
        if not has_cards_in_hand and (
            (self.current_player == 1 and not self.can_player1_collect)
            or (self.current_player == 2 and not self.can_player2_collect)
            or len(self.middle_cards) == 0
        ):
            actions.append(6)

        # print("actions:", actions)

        # There should always be at least one legal action
        # assert len(actions) > 0, "No legal actions available!"

        return actions

    def is_terminal(self) -> bool:
        """Check if the game is over"""
        # print(f"\nChecking terminal state:")
        # print(f"Deck size: {len(self.deck)}")
        # print(f"P1 hand: {len(self.player1_hand)}")
        # print(f"Middle cards: {len(self.middle_cards)}")
        # print(f"P1 can collect: {self.can_player1_collect}")
        # print(f"P2 can collect: {self.can_player2_collect}")
        # Game ends if either player has all city cards
        p1_cities = sum(
            1 for card in self.player1_collected if card.type == CardType.CITY
        )
        p2_cities = sum(
            1 for card in self.player2_collected if card.type == CardType.CITY
        )
        if p1_cities == 3 or p2_cities == 3:
            return True

        # Game also ends if no one can make any more moves
        player1_can_move = len(self.player1_hand) > 0 or (
            self.can_player1_collect and len(self.middle_cards) > 0
        )
        player2_can_move = self.player2_cards_played_this_round < 5 or (
            self.can_player2_collect and len(self.middle_cards) > 0
        )

        if not player1_can_move and not player2_can_move:
            return True

        return False

    def _get_unseen_card_features(self) -> List[float]:
        """Calculate features for tracking unseen cards"""
        features = []

        # Count visible cards
        visible_cards = (
            self.get_current_hand()  # Only your hand
            + self.middle_cards
            + self.player1_collected
            + self.player2_collected
            + self.removed_cards
        )

        # For each card type, calculate how many are still unseen
        for card_type in CardType:
            total = self.card_counts[card_type]
            visible = sum(1 for card in visible_cards if card.type == card_type)
            remaining = total - visible
            features.append(remaining / total)  # Normalize to [0,1]

        return features

    def make_move(self, action: int):
        """Execute a move in the game"""
        current_hand = self.get_current_hand()

        if action == 6:  # Pass (no cards in hand)
            pass  # Do nothing
        elif action == 5:  # Collect
            cards_to_collect = min(5, len(self.middle_cards))
            if self.current_player == 1:
                self.player1_collected.extend(self.middle_cards[-cards_to_collect:])
                self.can_player1_collect = False
            else:
                self.player2_collected.extend(self.middle_cards[-cards_to_collect:])
                self.can_player2_collect = False
            self.middle_cards = self.middle_cards[:-cards_to_collect]
        else:  # Play card
            if current_hand:
                if self.current_player == 1:
                    self.middle_cards.append(current_hand[action])
                    self.player1_hand.pop(action)
                else:
                    self.middle_cards.append(self.player2_hand[action])
                    self.player2_hand.pop(action)
                    self.player2_cards_played_this_round += 1

        # Deal new cards at end of round
        if (
            not self.player1_hand
            and self.player2_cards_played_this_round >= 5
            and not self.can_player1_collect
            and not self.can_player2_collect
            and self.deck
        ):
            self.player1_hand = self.deck[:5]
            self.deck = self.deck[5:]
            self.player2_hand = self.deck[:5]  # Deal to player 2
            self.deck = self.deck[5:]
            self.player2_cards_played_this_round = 0
            self.can_player1_collect = True
            self.can_player2_collect = True
            self.round_number += 1
            self.current_player = (
                self.first_player
                if self.round_number % 2 == 1
                else (3 - self.first_player)
            )
            return

        # Switch players
        self.current_player = 3 - self.current_player

    def get_current_hand(self) -> List[Card]:
        """Get the current player's hand"""
        if self.current_player == 1:
            return self.player1_hand
        else:
            return self.player2_hand

    def get_result(self) -> float:
        """Get the game result (1 for player 1 win, -1 for player 2 win, 0 for draw)"""
        # Check city card victory
        p1_cities = sum(
            1 for card in self.player1_collected if card.type == CardType.CITY
        )
        p2_cities = sum(
            1 for card in self.player2_collected if card.type == CardType.CITY
        )
        if p1_cities == 3:
            return 1.0
        if p2_cities == 3:
            return -1.0

        # Calculate points
        p1_score = self._calculate_score(self.player1_collected)
        p2_score = self._calculate_score(self.player2_collected)

        if p1_score > p2_score:
            return 1.0
        elif p2_score > p1_score:
            return -1.0
        return 0.0

    def clone(self) -> "GameState":
        """Create a deep copy of the game state"""
        new_state = GameState()
        new_state.deck = self.deck.copy()
        new_state.player1_hand = self.player1_hand.copy()
        new_state.player2_hand = self.player2_hand.copy()
        new_state.middle_cards = self.middle_cards.copy()
        new_state.player1_collected = self.player1_collected.copy()
        new_state.player2_collected = self.player2_collected.copy()
        new_state.removed_cards = self.removed_cards.copy()
        new_state.current_player = self.current_player
        new_state.can_player1_collect = self.can_player1_collect
        new_state.can_player2_collect = self.can_player2_collect
        new_state.round_number = self.round_number
        new_state.player2_cards_played_this_round = self.player2_cards_played_this_round
        return new_state

    def _calculate_score(self, collected_cards: List[Card]) -> int:
        """Calculate score for a player's collected cards"""
        score = 0

        # Count color majorities
        color_counts = {
            CardType.BLUE_5: 0,
            CardType.PINK_6: 0,
            CardType.ORANGE_7: 0,
            CardType.YELLOW_8: 0,
        }
        for card in collected_cards:
            if card.type in color_counts:
                color_counts[card.type] += 1

        # Score for color majorities
        other_collected = (
            self.player2_collected
            if collected_cards is self.player1_collected
            else self.player1_collected
        )
        other_counts = {
            CardType.BLUE_5: 0,
            CardType.PINK_6: 0,
            CardType.ORANGE_7: 0,
            CardType.YELLOW_8: 0,
        }
        for card in other_collected:
            if card.type in other_counts:
                other_counts[card.type] += 1

        # Add points for majorities
        if color_counts[CardType.BLUE_5] > other_counts[CardType.BLUE_5]:
            score += 5
        if color_counts[CardType.PINK_6] > other_counts[CardType.PINK_6]:
            score += 6
        if color_counts[CardType.ORANGE_7] > other_counts[CardType.ORANGE_7]:
            score += 7
        if color_counts[CardType.YELLOW_8] > other_counts[CardType.YELLOW_8]:
            score += 8

        # Score for complete sets (5-6-7-8)
        num_complete_sets = min(
            color_counts[CardType.BLUE_5],
            color_counts[CardType.PINK_6],
            color_counts[CardType.ORANGE_7],
            color_counts[CardType.YELLOW_8],
        )
        score += num_complete_sets * 5

        # Add modifier card values
        for card in collected_cards:
            if card.type == CardType.MOD_POS_2:
                score += 2
            elif card.type == CardType.MOD_POS_3:
                score += 3
            elif card.type == CardType.MOD_POS_4:
                score += 4
            elif card.type == CardType.MOD_NEG_1:
                score -= 1
            elif card.type == CardType.MOD_NEG_2:
                score -= 2
            elif card.type == CardType.MOD_NEG_3:
                score -= 3

        return score

    def pretty_print_state(self):
        """Pretty print the current game state"""
        print("\n" + "=" * 50)
        print(f"Round {self.round_number} - Player {self.current_player}'s turn")
        print("=" * 50)

        # Print role information
        print(f"\nRoles:")
        print(f"Player 1: {'Paperboy' if self.paperboy == 1 else 'Spy'}")
        print(f"Player 2: {'Paperboy' if self.paperboy == 2 else 'Spy'}")

        # Print middle cards with oldest → newest indication
        print("\nMiddle Cards (oldest → newest):")
        print("-" * 30)
        if not self.middle_cards:
            print("(empty)")
        else:
            for i, card in enumerate(self.middle_cards):
                if i == len(self.middle_cards) - 1:
                    print(f"{card.type.name} ← Most recent")
                else:
                    print(f"{card.type.name} → ", end="")

        # Print hands
        print("\nPlayer 1's Hand:")
        print("-" * 30)
        if not self.player1_hand:
            print("(empty)")
        else:
            for i, card in enumerate(self.player1_hand):
                print(f"{i}: {card.type.name}", end="  ")
        print("\nCan collect: " + ("Yes" if self.can_player1_collect else "No"))

        print("\nPlayer 2's Hand:")
        print("-" * 30)
        if not self.player2_hand:
            print("(empty)")
        else:
            for i, card in enumerate(self.player2_hand):
                print(f"{i}: {card.type.name}", end="  ")
        print("\nCan collect: " + ("Yes" if self.can_player2_collect else "No"))

        # Print collected cards
        print("\nCollected Cards:")
        print("-" * 30)
        print("Player 1:")
        if not self.player1_collected:
            print("(none)")
        else:
            # Group cards by type for cleaner display
            grouped = {}
            for card in self.player1_collected:
                grouped[card.type.name] = grouped.get(card.type.name, 0) + 1
            for card_type, count in sorted(grouped.items()):
                print(f"{card_type}: {count}", end="  ")

        print("\n\nPlayer 2:")
        if not self.player2_collected:
            print("(none)")
        else:
            grouped = {}
            for card in self.player2_collected:
                grouped[card.type.name] = grouped.get(card.type.name, 0) + 1
            for card_type, count in sorted(grouped.items()):
                print(f"{card_type}: {count}", end="  ")

        # Print scores
        print("\n\nCurrent Scores:")
        print("-" * 30)
        p1_score = self._calculate_score(self.player1_collected)
        p2_score = self._calculate_score(self.player2_collected)
        print(f"Player 1: {p1_score}")
        print(f"Player 2: {p2_score}")

        # Print city card count
        p1_cities = sum(
            1 for card in self.player1_collected if card.type == CardType.CITY
        )
        p2_cities = sum(
            1 for card in self.player2_collected if card.type == CardType.CITY
        )
        print(f"\nCity Cards - Player 1: {p1_cities}, Player 2: {p2_cities}")

        print("\n" + "=" * 50 + "\n")
