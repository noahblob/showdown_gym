import os
import time
from typing import Any, Dict, Iterable, Optional

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

TYPE_CHART = GenData.from_gen(9).type_chart

def _type_name(t) -> str:
        return (t.name if hasattr(t, "name") else str(t)).lower()

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
        self._last_action: Optional[int] = None 
    
    def _get_action_size(self) -> int | None:
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        try:
            self._last_action = int(action)
        except Exception:
            self._last_action = None
        return action

    def hint_action(self, battle: AbstractBattle) -> np.ndarray:
        onehot = np.zeros(10, dtype=np.float32)

        def calc_stab(my_types, opp) -> float:
            if not my_types or not opp:
                return 1.0
            best = 0.0
            for atk in my_types:
                best = max(best, opp.damage_multiplier(atk))
            return best

        def is_alive(mon) -> bool:
            return bool(mon and not getattr(mon, "fainted", False) and (getattr(mon, "current_hp", 0) > 0))

        me  = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        my_types  = me.types if me else []

        team_list = list(battle.team.values())[:6]
        valid_switch_idxs = []
        for i, mon in enumerate(team_list):
            if mon is None or mon is me:
                continue
            if is_alive(mon):
                valid_switch_idxs.append(i)

        avail_moves = battle.available_moves or []
        valid_moves = []
        for mi, mv in enumerate(avail_moves):
            if not bool(mv.disabled or False):
                valid_moves.append((6 + mi, mv))

        stay_best_a, stay_best_score, stay_best_eff = None, -1.0, 1.0
        for a, mv in valid_moves:
            bp = mv.base_power if hasattr(mv, "base_power") else 0.0
            mv_type = mv.type if hasattr(mv, 'type') else None
            
            eff = opp.damage_multiplier(mv_type) if (opp and mv_type) else 1.0
            
            if eff == 0.0:
                continue
            
            stab = calc_stab(my_types, opp)
            acc  = mv.accuracy if hasattr(mv, 'accuracy') and mv.accuracy else 100
            score = bp * eff * stab * acc
            if score > stay_best_score:
                stay_best_score = score
                stay_best_a = a
                stay_best_eff = eff

        if stay_best_a is None and valid_switch_idxs:
            best_idx, best_tuple = None, (-1e9, -1e9)
            for i in valid_switch_idxs:
                mon = team_list[i]
                my_to_opp  = calc_stab(mon.types, opp) if opp else 1.0
                opp_to_new = calc_stab(opp.types, mon) if opp else 1.0
                cand = (my_to_opp - opp_to_new, my_to_opp)
                if cand > best_tuple:
                    best_tuple, best_idx = cand, i
            if best_idx is not None:
                onehot[best_idx] = 1.0
                return onehot
            onehot[valid_switch_idxs[0]] = 1.0
            return onehot

        if stay_best_a is not None and stay_best_eff >= 1.0:
            onehot[stay_best_a] = 1.0
            return onehot

        best_sw_idx, best_sw_score = None, -1e9
        for i in valid_switch_idxs:
            mon = team_list[i]
            my_to_opp  = calc_stab(mon.types, opp) if opp else 1.0
            opp_to_new = calc_stab(opp.types, mon) if opp else 1.0
            sw_score = (my_to_opp) - (opp_to_new)
            if sw_score > best_sw_score:
                best_sw_score, best_sw_idx = sw_score, i


        if best_sw_idx is not None:
            if (stay_best_a is None) or (best_sw_score > 0.5):
                onehot[best_sw_idx] = 1.0
                return onehot

        if stay_best_a is not None:
            onehot[stay_best_a] = 1.0
            return onehot

        if valid_switch_idxs:
            onehot[valid_switch_idxs[0]] = 1.0
        elif valid_moves:
            onehot[valid_moves[0][0]] = 1.0
        return onehot

    def _observation_size(self) -> int:
        return 10

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        onehot = self.hint_action(battle).astype(np.float32)
        return onehot

    def calc_reward(self, battle: AbstractBattle) -> float:
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None

        if prior is None:
            return 0.0 

        hint_prior = self.hint_action(prior)
        hinted_idx = int(np.argmax(hint_prior)) if hint_prior.sum() > 0 else None

        if hinted_idx is not None and self._last_action is not None:
            return 1.0 if int(self._last_action) == hinted_idx else 0.0

        return 0.0


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################

class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.
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
            opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
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