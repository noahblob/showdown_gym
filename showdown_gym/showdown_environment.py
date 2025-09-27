import os
import time
from typing import Any, Dict

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player
from poke_env.data.gen_data import GenData

from showdown_gym.base_environment import BaseShowdownEnv


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        This reward function is designed to beat max damage bots by incentivizing:
        1. Dealing damage while minimizing damage taken
        2. Winning battles with large bonuses
        3. Keeping Pokemon alive for strategic advantage
        4. Making progress toward victory

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)
        reward = 0.0

        # Battle outcome rewards (highest priority)
        if battle.battle_tag and battle.finished:
            if battle.won:
                reward += 10.0  # Large reward for winning
            else:
                reward -= 5.0  # Penalty for losing
            return reward

        # Only calculate incremental rewards if we have a prior state
        if prior_battle is None:
            return 0.0

        # Get current and prior health states
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        prior_health_team = [
            mon.current_hp_fraction for mon in prior_battle.team.values()
        ]
        prior_health_opponent = [
            mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
        ]

        # Ensure consistent array lengths (pad with 1.0 for missing Pokemon)
        max_team_size = max(len(health_team), len(prior_health_team))
        health_team.extend([1.0] * (max_team_size - len(health_team)))
        prior_health_team.extend([1.0] * (max_team_size - len(prior_health_team)))

        max_opp_size = max(len(health_opponent), len(prior_health_opponent))
        health_opponent.extend([1.0] * (max_opp_size - len(health_opponent)))
        prior_health_opponent.extend(
            [1.0] * (max_opp_size - len(prior_health_opponent))
        )

        # Calculate health changes
        team_damage_taken = np.sum(np.array(prior_health_team) - np.array(health_team))
        opponent_damage_dealt = np.sum(
            np.array(prior_health_opponent) - np.array(health_opponent)
        )

        # Reward damage dealt to opponent (positive)
        reward += opponent_damage_dealt * 2.0

        # Penalty for damage taken (negative, but smaller magnitude to encourage aggression)
        reward -= team_damage_taken * 1.0

        # Bonus for favorable damage trades (dealt more than received)
        if opponent_damage_dealt > team_damage_taken:
            reward += (opponent_damage_dealt - team_damage_taken) * 0.5

        # Count Pokemon fainted (KO bonuses/penalties)
        team_fainted = sum(1 for hp in health_team if hp == 0) - sum(
            1 for hp in prior_health_team if hp == 0
        )
        opp_fainted = sum(1 for hp in health_opponent if hp == 0) - sum(
            1 for hp in prior_health_opponent if hp == 0
        )

        reward += opp_fainted * 3.0  # Large bonus for KO'ing opponent Pokemon
        reward -= team_fainted * 2.0  # Penalty for losing Pokemon

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        The observation includes:
        - 6 features: our team health fractions
        - 6 features: opponent team health fractions
        - 8 strategic features: speed advantage, type effectiveness, resistance, move power,
          status conditions, and team sizes
        Total: 6 + 6 + 8 = 20 features

        Returns:
            int: The size of the observation space (20).
        """

        return 20

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.

        This embedding provides strategic information to help beat max damage bots:
        - Health information for decision making
        - Type effectiveness for resistance/advantage
        - Speed comparison for turn order decisions
        - Active Pokemon information for switching decisions

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the strategic battle state features.
        """
        
        type_chart = GenData.from_gen(9).type_chart

        # Basic health information
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Pad health arrays to ensure consistent size (assuming max 6 Pokemon)
        health_team.extend([0.0] * (6 - len(health_team)))
        health_opponent.extend([0.0] * (6 - len(health_opponent)))

        # Active Pokemon information
        active_mon = battle.active_pokemon
        opp_active_mon = battle.opponent_active_pokemon

        # Speed comparison (normalized) - helps with switching decisions
        speed_advantage = 0.0
        if active_mon and opp_active_mon and opp_active_mon.stats.get("spe"):
            our_speed = active_mon.stats.get("spe", 100)
            opp_speed = opp_active_mon.stats.get("spe", 100)
            # Normalize speed difference to [-1, 1] range
            speed_advantage = min(max((our_speed - opp_speed) / 200.0, -1.0), 1.0)

        # Type effectiveness of our active Pokemon against opponent
        type_advantage = 0.0
        if active_mon and opp_active_mon:
            # Calculate average type effectiveness of our types against opponent
            our_types = active_mon.types
            opp_types = opp_active_mon.types
            if our_types and opp_types:
                effectiveness_values = []
                for our_type in our_types:
                    for opp_type in opp_types:
                        # This is a simplified type effectiveness calculation
                        # In a real implementation, you'd want to use the actual type chart
                        effectiveness = our_type.damage_multiplier(opp_type, type_chart=type_chart)
                        effectiveness_values.append(effectiveness)
                if effectiveness_values:
                    # Convert to log scale and normalize: 0.25->-1, 0.5->-0.5, 1->0, 2->0.5, 4->1
                    avg_effectiveness = np.mean(effectiveness_values)
                    type_advantage = (
                        min(max(np.log2(avg_effectiveness), -2.0), 2.0) / 2.0
                    )

        # Type resistance (how well we resist opponent's attacks)
        type_resistance = 0.0
        if active_mon and opp_active_mon:
            # Calculate how resistant we are to opponent's types
            resistance_values = []
            for opp_type in opp_active_mon.types:
                for our_type in active_mon.types:
                    resistance = opp_type.damage_multiplier(our_type, type_chart=type_chart)
                    resistance_values.append(resistance)
            if resistance_values:
                avg_resistance = np.mean(resistance_values)
                # Invert and normalize: 4x damage taken -> -1, 2x -> -0.5, 1x -> 0, 0.5x -> 0.5, 0.25x -> 1
                type_resistance = min(max(-np.log2(avg_resistance), -2.0), 2.0) / 2.0

        # Available moves power (normalized average)
        avg_move_power = 0.0
        if active_mon and active_mon.moves:
            move_powers = []
            for move in active_mon.moves.values():
                if move.base_power:
                    move_powers.append(move.base_power)
            if move_powers:
                avg_move_power = min(
                    np.mean(move_powers) / 120.0, 1.0
                )  # Normalize to [0,1]

        # Status condition indicators
        our_status = 0.0
        opp_status = 0.0
        if active_mon:
            our_status = 1.0 if active_mon.status else 0.0
        if opp_active_mon:
            opp_status = 1.0 if opp_active_mon.status else 0.0

        # Team size remaining (normalized)
        team_alive = sum(1 for hp in health_team if hp > 0) / 6.0
        opp_alive = sum(1 for hp in health_opponent if hp > 0) / 6.0

        # Combine all features
        strategic_features = [
            speed_advantage,  # -1 to 1: negative means opponent is faster
            type_advantage,  # -1 to 1: positive means our attacks are effective
            type_resistance,  # -1 to 1: positive means we resist their attacks
            avg_move_power,  # 0 to 1: normalized average move power
            our_status,  # 0 or 1: whether we have status condition
            opp_status,  # 0 or 1: whether opponent has status condition
            team_alive,  # 0 to 1: fraction of our team alive
            opp_alive,  # 0 to 1: fraction of opponent team alive
        ]

        # Final vector combines health info and strategic features
        final_vector = np.concatenate(
            [
                health_team,  # 6 features: our team health
                health_opponent,  # 6 features: opponent team health
                strategic_features,  # 8 features: strategic information
            ]
        )

        return np.array(final_vector, dtype=np.float32)


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
