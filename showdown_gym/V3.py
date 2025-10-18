import os
import time
from typing import Any, Dict, List, Optional

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
        self.allowed_actions = list(range(-2, 10)) + list(range(22, 26))
        
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one
        self._type_chart = GenData.from_gen(9).type_chart
        self._reset_tracking_state()

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return len(self.allowed_actions)  # Return None if action size is default

    def reset(self, seed: Any | None = None, options: Dict[str, Any] | None = None):
        response = super().reset(seed=seed, options=options)
        self._reset_tracking_state()
        return response

    def _reset_tracking_state(self) -> None:
        self._last_action_index = None
        self._last_action_value = None
        self._last_step_time_ms = 0.0
        self._cumulative_reward = 0.0
        self._last_reward_components = self._init_reward_components()
        self._last_step_reward = 0.0

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
        idx = int(action)
        if idx < 0 or idx >= len(self.allowed_actions):
            raise ValueError(f"Invalid action index: {idx}")
        action_value = self.allowed_actions[idx]
        self._last_action_index = idx
        self._last_action_value = action_value
        return np.int64(action_value)

    def step(
        self, actions: dict[str, np.int64]
    ) -> tuple[
        dict[str, ObsType],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        start = time.perf_counter()
        observations, rewards, terminated, truncated, info = super().step(actions)
        self._last_step_time_ms = (time.perf_counter() - start) * 1000.0
        agent_key = self.possible_agents[0]
        self._cumulative_reward += float(rewards.get(agent_key, 0.0))
        return observations, rewards, terminated, truncated, info

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        if self.battle1 is None:
            return info

        battle = self.battle1
        agent = self.possible_agents[0]
        agent_info = info.setdefault(agent, {})
        agent_info["win"] = battle.won

        strategic = self._compute_strategic_features(battle)

        team_hp_vector = self._health_vector(battle.team.values(), 6, 0.0)
        opp_hp_vector = self._health_vector(battle.opponent_team.values(), 6, 0.0)

        team_size = max(len(battle.team), 1)
        opp_size = max(len(battle.opponent_team), 1)

        team_alive_count = sum(
            1 for mon in battle.team.values() if mon.current_hp_fraction > 0.0
        )
        opp_alive_count = sum(
            1 for mon in battle.opponent_team.values() if mon.current_hp_fraction > 0.0
        )

        team_alive_fraction = team_alive_count / team_size
        opp_alive_fraction = opp_alive_count / opp_size

        fainted_diff = (opp_size - opp_alive_count) - (team_size - team_alive_count)

        total_team_hp = float(
            sum(mon.current_hp_fraction for mon in battle.team.values())
        )
        total_opp_hp = float(
            sum(mon.current_hp_fraction for mon in battle.opponent_team.values())
        )

        agent_info.update(
            {
                "turn": battle.turn,
                "battle_tag": battle.battle_tag,
                "battle_format": getattr(self, "format", None),
                "finished": battle.finished,
                "result_reason": self._stringify_state(
                    getattr(battle, "end_reason", None)
                    or getattr(battle, "_end_reason", None)
                ),
                "win_prob_estimate": self._estimate_win_probability(
                    total_team_hp, total_opp_hp
                ),
                "team_hp": team_hp_vector,
                "opp_hp": opp_hp_vector,
                "team_alive": team_alive_fraction,
                "opp_alive": opp_alive_fraction,
                "team_alive_count": team_alive_count,
                "opp_alive_count": opp_alive_count,
                "fainted_diff": fainted_diff,
                "last_action": {
                    "index": self._last_action_index,
                    "resolved_action": self._last_action_value,
                    "description": self._describe_action_value(
                        self._last_action_value
                    ),
                },
                "legal_actions": self._gather_legal_actions(battle),
                "switch_required": bool(getattr(battle, "force_switch", False)),
                "our_statuses": self._gather_statuses(battle.team.values()),
                "opp_statuses": self._gather_statuses(
                    battle.opponent_team.values()
                ),
                "field_states": self._collect_field_states(battle),
                "reward_components": dict(self._last_reward_components),
                "cumulative_reward": self._cumulative_reward,
                "last_reward": self._last_step_reward,
                "speed_advantage": strategic["speed_advantage"],
                "type_advantage": strategic["type_advantage"],
                "type_resistance": strategic["type_resistance"],
                "avg_move_power": strategic["avg_move_power"],
                "our_active_status": strategic["our_status"],
                "opp_active_status": strategic["opp_status"],
                "team_alive_fraction": strategic["team_alive_fraction"],
                "opp_alive_fraction": strategic["opp_alive_fraction"],
                "strategic_features": dict(strategic),
                "step_time_ms": self._last_step_time_ms,
                "epsilon": getattr(self, "epsilon", None),
            }
        )

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
        components = self._init_reward_components()

        # Battle outcome rewards (highest priority)
        if battle.battle_tag and battle.finished:
            if battle.won:
                components["win_bonus"] = 25.0
            else:
                components["loss_penalty"] = -20.0
            reward = components["win_bonus"] + components["loss_penalty"]
            self._update_reward_state(battle, reward, components)
            return reward

        # Only calculate incremental rewards if we have a prior state
        if prior_battle is None:
            self._update_reward_state(battle, reward, components)
            return reward

        # Get current and prior health states and pad for consistent lengths
        current_team = self._health_vector(
            battle.team.values(),
            max(len(battle.team), len(prior_battle.team)),
            1.0,
        )
        prior_team = self._health_vector(
            prior_battle.team.values(),
            max(len(battle.team), len(prior_battle.team)),
            1.0,
        )

        current_opponent = self._health_vector(
            battle.opponent_team.values(),
            max(len(battle.opponent_team), len(prior_battle.opponent_team)),
            1.0,
        )
        prior_opponent = self._health_vector(
            prior_battle.opponent_team.values(),
            max(len(battle.opponent_team), len(prior_battle.opponent_team)),
            1.0,
        )

        team_damage_taken = float(
            np.sum(np.array(prior_team) - np.array(current_team))
        )
        opponent_damage_dealt = float(
            np.sum(np.array(prior_opponent) - np.array(current_opponent))
        )

        components["opponent_damage_dealt_reward"] = opponent_damage_dealt * 2.0
        reward += components["opponent_damage_dealt_reward"]

        components["team_damage_taken_penalty"] = -team_damage_taken * 1.0
        reward += components["team_damage_taken_penalty"]

        if opponent_damage_dealt > team_damage_taken:
            components["damage_trade_bonus"] = (
                opponent_damage_dealt - team_damage_taken
            ) * 0.5
            reward += components["damage_trade_bonus"]

        team_fainted_delta = sum(1 for hp in current_team if hp == 0.0) - sum(
            1 for hp in prior_team if hp == 0.0
        )
        opp_fainted_delta = sum(1 for hp in current_opponent if hp == 0.0) - sum(
            1 for hp in prior_opponent if hp == 0.0
        )

        components["opponent_fainted_bonus"] = opp_fainted_delta * 3.0
        reward += components["opponent_fainted_bonus"]

        components["team_fainted_penalty"] = -team_fainted_delta * 2.0
        reward += components["team_fainted_penalty"]

        self._update_reward_state(battle, reward, components)
        return reward

    @staticmethod
    def _init_reward_components() -> Dict[str, float]:
        return {
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "opponent_damage_dealt_reward": 0.0,
            "team_damage_taken_penalty": 0.0,
            "damage_trade_bonus": 0.0,
            "opponent_fainted_bonus": 0.0,
            "team_fainted_penalty": 0.0,
        }

    def _update_reward_state(
        self,
        battle: AbstractBattle,
        reward: float,
        components: Dict[str, float],
    ) -> None:
        if self.battle1 is battle:
            self._last_reward_components = {
                key: float(value) for key, value in components.items()
            }
            self._last_step_reward = float(reward)

    @staticmethod
    def _health_vector(source: Any, target_len: int, pad_value: float) -> List[float]:
        values: List[float] = []
        for element in source:
            if hasattr(element, "current_hp_fraction"):
                values.append(float(getattr(element, "current_hp_fraction", 0.0)))
            else:
                values.append(float(element))
        if target_len > 0 and len(values) > target_len:
            values = values[:target_len]
        if target_len > len(values):
            values.extend([float(pad_value)] * (target_len - len(values)))
        return values

    def _compute_strategic_features(self, battle: AbstractBattle) -> Dict[str, float]:
        active_mon = battle.active_pokemon
        opp_active_mon = battle.opponent_active_pokemon

        speed_advantage = 0.0
        if active_mon and opp_active_mon and getattr(opp_active_mon.stats, "spe", None):
            our_speed = float(active_mon.stats.get("spe", 100))
            opp_speed = float(opp_active_mon.stats.get("spe", 100))
            speed_advantage = float(
                max(min((our_speed - opp_speed) / 200.0, 1.0), -1.0)
            )

        type_advantage = 0.0
        type_resistance = 0.0
        if active_mon and opp_active_mon:
            our_types = getattr(active_mon, "types", None)
            opp_types = getattr(opp_active_mon, "types", None)
            if our_types and opp_types:
                effectiveness_values: List[float] = []
                resistance_values: List[float] = []
                for our_type in our_types:
                    for opp_type in opp_types:
                        effectiveness = float(
                            our_type.damage_multiplier(
                                opp_type, type_chart=self._type_chart
                            )
                        )
                        effectiveness_values.append(effectiveness)
                        resistance = float(
                            opp_type.damage_multiplier(
                                our_type, type_chart=self._type_chart
                            )
                        )
                        resistance_values.append(resistance)
                if effectiveness_values:
                    avg_effectiveness = float(np.mean(effectiveness_values))
                    if avg_effectiveness > 0:
                        type_advantage = float(
                            max(min(np.log2(avg_effectiveness), 2.0), -2.0) / 2.0
                        )
                if resistance_values:
                    avg_resistance = float(np.mean(resistance_values))
                    if avg_resistance > 0:
                        type_resistance = float(
                            max(min(-np.log2(avg_resistance), 2.0), -2.0) / 2.0
                        )

        avg_move_power = 0.0
        if active_mon and getattr(active_mon, "moves", None):
            move_powers = [
                float(move.base_power)
                for move in active_mon.moves.values()
                if getattr(move, "base_power", None)
            ]
            if move_powers:
                avg_move_power = float(min(np.mean(move_powers) / 120.0, 1.0))

        our_status = 1.0 if active_mon and getattr(active_mon, "status", None) else 0.0
        opp_status = (
            1.0 if opp_active_mon and getattr(opp_active_mon, "status", None) else 0.0
        )

        team_health = [mon.current_hp_fraction for mon in battle.team.values()]
        opp_health = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
        team_health.extend([0.0] * (6 - len(team_health)))
        opp_health.extend([0.0] * (6 - len(opp_health)))

        team_alive_fraction = sum(1 for hp in team_health if hp > 0.0) / 6.0
        opp_alive_fraction = sum(1 for hp in opp_health if hp > 0.0) / 6.0

        return {
            "speed_advantage": speed_advantage,
            "type_advantage": type_advantage,
            "type_resistance": type_resistance,
            "avg_move_power": avg_move_power,
            "our_status": our_status,
            "opp_status": opp_status,
            "team_alive_fraction": team_alive_fraction,
            "opp_alive_fraction": opp_alive_fraction,
        }

    @staticmethod
    def _stringify_state(value: Any) -> Optional[str]:
        if value is None:
            return None
        if hasattr(value, "name"):
            return str(getattr(value, "name"))
        return str(value)

    def _describe_action_value(self, action_value: Optional[int]) -> Optional[str]:
        if action_value is None:
            return None
        if action_value == -2:
            return "default"
        if action_value == -1:
            return "forfeit"
        if 0 <= action_value <= 5:
            return f"switch_{action_value}"
        if 6 <= action_value <= 9:
            return f"move_{action_value - 5}"
        if 10 <= action_value <= 13:
            return f"mega_move_{action_value - 9}"
        if 14 <= action_value <= 17:
            return f"z_move_{action_value - 13}"
        if 18 <= action_value <= 21:
            return f"dynamax_move_{action_value - 17}"
        if 22 <= action_value <= 25:
            return f"terastallize_move_{action_value - 21}"
        return str(action_value)

    def _gather_legal_actions(self, battle: AbstractBattle) -> Dict[str, Any]:
        moves_info: List[Dict[str, Any]] = []
        for move in getattr(battle, "available_moves", None) or []:
            moves_info.append(
                {
                    "id": getattr(move, "id", None),
                    "name": getattr(move, "name", getattr(move, "id", None)),
                    "base_power": getattr(move, "base_power", None),
                    "current_pp": getattr(move, "current_pp", None),
                    "max_pp": getattr(move, "max_pp", None),
                    "disabled": getattr(move, "disabled", False),
                }
            )

        switches_info: List[Dict[str, Any]] = []
        for pokemon in getattr(battle, "available_switches", None) or []:
            switches_info.append(
                {
                    "species": getattr(pokemon, "species", None),
                    "status": self._stringify_state(getattr(pokemon, "status", None)),
                    "hp_fraction": float(getattr(pokemon, "current_hp_fraction", 0.0)),
                }
            )

        def _as_bool(candidate: Any) -> bool:
            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    return True
            return bool(candidate)

        return {
            "moves": moves_info,
            "switches": switches_info,
            "can_mega_evolve": _as_bool(getattr(battle, "can_mega_evolve", False)),
            "can_z_move": _as_bool(getattr(battle, "can_z_move", False)),
            "can_dynamax": _as_bool(getattr(battle, "can_dynamax", False)),
            "can_terastallize": _as_bool(getattr(battle, "can_tera", False)),
        }

    def _gather_statuses(self, mons: Any) -> List[Dict[str, Any]]:
        statuses: List[Dict[str, Any]] = []
        for mon in mons:
            boosts = getattr(mon, "boosts", None)
            statuses.append(
                {
                    "species": getattr(mon, "species", None),
                    "status": self._stringify_state(getattr(mon, "status", None)),
                    "hp_fraction": float(getattr(mon, "current_hp_fraction", 0.0)),
                    "boosts": dict(boosts) if isinstance(boosts, dict) else None,
                }
            )
        return statuses

    def _collect_field_states(self, battle: AbstractBattle) -> Dict[str, Any]:
        fields = getattr(battle, "fields", {})
        side_conditions = getattr(battle, "side_conditions", {})
        opp_side_conditions = getattr(battle, "opponent_side_conditions", {})

        field_list: List[Optional[str]]
        if isinstance(fields, dict):
            field_list = [
                self._stringify_state(field) for field, turns in fields.items() if turns
            ]
        else:
            field_list = [self._stringify_state(field) for field in fields]

        side_conditions_list: List[Optional[str]]
        if isinstance(side_conditions, dict):
            side_conditions_list = [
                self._stringify_state(condition)
                for condition, turns in side_conditions.items()
                if turns
            ]
        else:
            side_conditions_list = [
                self._stringify_state(condition) for condition in side_conditions
            ]

        opp_side_conditions_list: List[Optional[str]]
        if isinstance(opp_side_conditions, dict):
            opp_side_conditions_list = [
                self._stringify_state(condition)
                for condition, turns in opp_side_conditions.items()
                if turns
            ]
        else:
            opp_side_conditions_list = [
                self._stringify_state(condition) for condition in opp_side_conditions
            ]

        return {
            "weather": self._stringify_state(getattr(battle, "weather", None)),
            "terrain": self._stringify_state(getattr(battle, "terrain", None)),
            "fields": field_list,
            "side_conditions": side_conditions_list,
            "opponent_side_conditions": opp_side_conditions_list,
        }

    @staticmethod
    def _estimate_win_probability(team_hp_total: float, opp_hp_total: float) -> float:
        denominator = team_hp_total + opp_hp_total
        if denominator <= 1e-6:
            return 0.5
        return float(max(min(team_hp_total / denominator, 1.0), 0.0))

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

        health_team = self._health_vector(battle.team.values(), 6, 0.0)
        health_opponent = self._health_vector(battle.opponent_team.values(), 6, 0.0)

        strategic = self._compute_strategic_features(battle)
        strategic_features = [
            strategic["speed_advantage"],
            strategic["type_advantage"],
            strategic["type_resistance"],
            strategic["avg_move_power"],
            strategic["our_status"],
            strategic["opp_status"],
            strategic["team_alive_fraction"],
            strategic["opp_alive_fraction"],
        ]

        final_vector = np.concatenate(
            [
                np.array(health_team, dtype=np.float32),
                np.array(health_opponent, dtype=np.float32),
                np.array(strategic_features, dtype=np.float32),
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
