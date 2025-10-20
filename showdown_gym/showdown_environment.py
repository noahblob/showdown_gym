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
from poke_env.environment.singles_env import ObsType
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
        self._last_action: int | None = None
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        try:
            self._last_action = int(action)
        except Exception:
            self._last_action = None
        
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def hint_action(self, battle: AbstractBattle):
        """
        Uses a simple heuristic to suggest an action for the agent.
        - Calculate most effective move of current pokemon using base_power * type_effectiveness * stab * weather_mult * terrain_mult * accuracy_mult
        - Check if there is a better switch if effectiveness of current move is < 1
        - Switch if there is a better option, e.g. effectiveness > 1
        - If an index in the output array is 1, that action is suggested.
        - Else if index is 0, that action is not suggested.
        - First 6 indices are switches, next 4 incides are moves (6 possible switches, 4 moves max, maps nicely to action space)
        """
        type_chart = GenData.from_gen(9).type_chart

        output = np.zeros(self._get_action_size(), dtype=np.int8)
        
        # Get current pokemon and opponent
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        
        if me is None or opp is None:
            return output
        
        my_types = me.types
        opp_types = opp.types
        
        # Helper function to calculate type effectiveness
        def calc_effectiveness(move_type, defender_types):
            """Calculate type effectiveness multiplier"""
            if move_type is None:
                return 1.0
            # Convert move_type to string key
            if hasattr(move_type, 'name'):
                move_type_str = move_type.name.lower()
            else:
                move_type_str = str(move_type).lower()
            multiplier = 1.0
            for def_type in defender_types:
                if def_type is not None:
                    if hasattr(def_type, 'name'):
                        def_type_str = def_type.name.lower()
                    else:
                        def_type_str = str(def_type).lower()
                    try:
                        effectiveness = type_chart[move_type_str][def_type_str]
                    except KeyError:
                        # fallback: try stripping extra info or use 1.0
                        effectiveness = 1.0
                    multiplier *= effectiveness
            return multiplier
        
        # Helper function to check move immunity due to ability
        def is_immune_ability(move_type, ability):
            """Check if move is immune due to opponent's ability"""
            if not ability:
                return False
            ability_name = str(ability).lower()
            immunity_map = {
                "levitate": "ground",
                "flashfire": "fire",
                "waterabsorb": "water",
                "dryskin": "water",
                "sapsipper": "grass",
                "lightningrod": "electric",
                "stormdrain": "water",
                "voltabsorb": "electric",
            }
            if hasattr(move_type, 'name'):
                move_type_name = move_type.name.lower()
            else:
                move_type_name = str(move_type).lower() or ""
            return immunity_map.get(ability_name) == move_type_name
        
        # Calculate weather and terrain boosts
        weather_boost_map = {}
        terrain_boost_map = {}
        
        # Check for weather effects
        if hasattr(battle, "weather"):
            weather = battle.weather
            if weather:
                weather_name = str(weather).lower()
                if "sunnyday" in weather_name or "sun" in weather_name:
                    weather_boost_map["fire"] = 1.5
                    weather_boost_map["water"] = 0.5
                elif "raindance" in weather_name or "rain" in weather_name:
                    weather_boost_map["water"] = 1.5
                    weather_boost_map["fire"] = 0.5
        
        # Check for terrain effects
        if hasattr(battle, "fields"):
            for field in battle.fields:
                field_name = str(field).lower()
                if "electricterrain" in field_name:
                    terrain_boost_map["electric"] = 1.3
                elif "grassyterrain" in field_name:
                    terrain_boost_map["grass"] = 1.3
                elif "psychicterrain" in field_name:
                    terrain_boost_map["psychic"] = 1.3
                elif "mistyterrain" in field_name:
                    terrain_boost_map["dragon"] = 0.5
        
        # Evaluate available moves
        best_move_idx = None
        best_move_score = -1.0
        best_move_effectiveness = 0.0
        
        for i, move in enumerate(battle.available_moves):
            if i >= 4:  # Only consider first 4 moves
                break
            
            move_type = move.type
            base_power = move.base_power if move.base_power else 0
            
            if base_power <= 0:
                continue
            
            # Check ability immunity
            opp_ability = opp.ability if hasattr(opp, 'ability') else None
            if is_immune_ability(move_type, opp_ability):
                continue
            
            # Calculate type effectiveness
            type_effectiveness = calc_effectiveness(move_type, opp_types)
            
            if type_effectiveness == 0.0:
                continue
            
            # STAB bonus
            stab = 1.5 if move_type in my_types else 1.0
            
            # Weather boost
            if hasattr(move_type, 'name'):
                move_type_name = move_type.name.lower()
            else:
                move_type_name = str(move_type).lower() or ""
            weather_mult = weather_boost_map.get(move_type_name, 1.0)
            
            # Terrain boost
            terrain_mult = terrain_boost_map.get(move_type_name, 1.0)
            
            # Accuracy factor
            accuracy = move.accuracy if hasattr(move, 'accuracy') and move.accuracy else 100
            accuracy_mult = accuracy / 100.0 if accuracy else 1.0
            
            # Calculate total score
            total_score = base_power * type_effectiveness * stab * weather_mult * terrain_mult * accuracy_mult
            
            if total_score > best_move_score:
                best_move_score = total_score
                best_move_idx = i
                best_move_effectiveness = type_effectiveness
        
        # Evaluate potential switches
        best_switch_idx = None
        best_switch_score = -1.0
        
        team_list = list(battle.team.values())
        for i, mon in enumerate(team_list):
            if i >= 6:  # Only consider first 6 pokemon
                break
            
            # Skip if this is the active pokemon or if fainted
            if mon == me or mon.fainted or mon not in battle.available_switches:
                continue
            
            switch_types = mon.types
            
            # Calculate offensive advantage (how well switch-in hits opponent)
            offensive_advantage = 0.0
            for switch_type in switch_types:
                if switch_type is not None:
                    eff = calc_effectiveness(switch_type, opp_types)
                    offensive_advantage = max(offensive_advantage, eff)
            
            # Calculate defensive risk (how well opponent hits switch-in)
            defensive_risk = 1.0
            for opp_type in opp_types:
                if opp_type is not None:
                    eff = calc_effectiveness(opp_type, switch_types)
                    defensive_risk = max(defensive_risk, eff)
            
            # Score: prioritize offensive advantage and minimize defensive risk
            switch_score = (offensive_advantage * 2.0) / (defensive_risk + 0.5)
            
            if switch_score > best_switch_score:
                best_switch_score = switch_score
                best_switch_idx = i
        
        # Decision logic: switch if current move is not effective and we have a better switch
        if best_move_effectiveness < 1.0 and best_switch_idx is not None and best_switch_score > 1.5:
            # Suggest the best switch
            output[best_switch_idx] = 1
        elif best_move_idx is not None:
            # Suggest the best move (indices 6-9 for moves)
            output[6 + best_move_idx] = 1
        elif best_switch_idx is not None:
            # If no good moves, suggest best switch
            output[best_switch_idx] = 1
        else:
            # Fallback: suggest first available action
            if battle.available_moves:
                output[6] = 1
            elif battle.available_switches:
                output[0] = 1
        
        return output

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        Pure hint imitation reward:
        +1.0 if the chosen action equals the hint computed on the PRIOR state
        +0.0 otherwise

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
        Returns:
            float: The calculated reward based on hint alignment.
        """
        if battle.finished:
            return 10.0 if battle.won else -5.0

        try:
            prior_battle = self._get_prior_battle(battle)
        except AttributeError:
            prior_battle = None

        if prior_battle is None:
            return 0.0

        # Get hint action from prior state
        previous_hint = self.hint_action(prior_battle)
        hinted_idx = int(np.argmax(previous_hint)) if previous_hint.sum() > 0 else None

        # Reward if agent followed the hint
        if hinted_idx is not None and hasattr(self, '_last_action') and self._last_action is not None:
            return 1.0 if int(self._last_action) == hinted_idx else 0.0

        return 0.0

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        return 10

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        Returns a one-hot encoded hint action array where exactly one index is 1:
        - Indices 0-5: switch to pokemon 0-5
        - Indices 6-9: use move 0-3

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the hint action one-hot vector (length 10).
        """
        
        # Get hint action one-hot vector (10 values: 0-5 for switches, 6-9 for moves)
        hint = self.hint_action(battle).astype(np.float32)
        
        return hint


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