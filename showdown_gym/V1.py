import os
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Defensive imports for enums/constants that may differ across poke-env versions
try:
    from poke_env.environment.field import Field
    from poke_env.environment.side_condition import SideCondition
    from poke_env.environment.weather import Weather
except Exception:
    Field = object  # type: ignore
    SideCondition = object  # type: ignore
    Weather = object  # type: ignore

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

        # Cache Gen 9 data for mappings (species, moves, items, abilities, types)
        self._gen_data = GenData.from_gen(9)
        self._species_index = {k.lower(): i for i, k in enumerate(sorted(self._gen_data.species))}
        self._ability_index = {k.lower(): i for i, k in enumerate(sorted(self._gen_data.abilities))}
        self._item_index = {k.lower(): i for i, k in enumerate(sorted(self._gen_data.items))}
        self._move_index = {k.lower(): i for i, k in enumerate(sorted(self._gen_data.moves))}
        self._types = sorted(self._gen_data.type_chart.keys())
        # Some repos use "snow" replacing "hail" in gen9; we still encode into the hail channel.
        self._type_to_idx = {t.lower(): i for i, t in enumerate(self._types)}

        # Volatile effects list (38 entries). If an effect name here is present on the Pokémon,
        # we mark it ON. Fill to 38 if your list changes.
        self._volatile_names: List[str] = [
            # common general-purpose volatiles
            "confusion", "attract", "substitute", "leechseed", "perishsong", "ingrain", "aqua ring",
            "curse", "disable", "encore", "taunt", "torment", "embargo", "imprison", "nightmare",
            "yawn", "telekinesis", "healblock", "foresight", "miracle eye", "magnet rise",
            "power trick", "trap", "partialtrap", "spite", "flashfire", "slowstart", "drowsy",
            "focusenergy", "stockpile1", "stockpile2", "stockpile3", "substitutebroken",
            # two-turn move preparations (preparing)
            "fly", "bounce", "dig", "dive", "shadowforce", "phantomforce",
        ]
        # pad to exactly 38 entries if necessary
        if len(self._volatile_names) < 38:
            self._volatile_names += [f"custom_{i}" for i in range(38 - len(self._volatile_names))]
        elif len(self._volatile_names) > 38:
            self._volatile_names = self._volatile_names[:38]

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return len(self.allowed_actions)  # Return None if action size is default

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
        return np.int64(self.allowed_actions[idx])

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
                reward += 25  # Large reward for winning
            else:
                reward -= 20  # Penalty for losing
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

    # =======================
    # Appendix A encoders
    # =======================

    def _observation_size(self) -> int:
        # 125 non-Pokémon features + 12 * 300 per-Pokémon features
        return 125 + 12 * 300

    @staticmethod
    def _one_hot(index: int, length: int) -> np.ndarray:
        v = np.zeros(length, dtype=np.float32)
        if 0 <= index < length:
            v[index] = 1.0
        return v

    @staticmethod
    def _presence_one_hot(present: bool) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32) if not present else np.array([0.0, 1.0], dtype=np.float32)

    @staticmethod
    def _bin_hp_fraction(hp_frac: float) -> np.ndarray:
        # 7 bins: bin0 for 0 HP (fainted), bins 1..6 for (0,1] equally sized
        if hp_frac <= 0.0:
            return ShowdownEnvironment._one_hot(0, 7)
        # Map (0,1] -> 1..6
        idx = int(np.ceil(np.clip(hp_frac, 1e-6, 1.0) * 6.0))
        idx = int(np.clip(idx, 1, 6))
        return ShowdownEnvironment._one_hot(idx, 7)

    @staticmethod
    def _get_dict_value(d: Any, key: Any, default: int = 0) -> int:
        try:
            if hasattr(d, "get"):
                return int(d.get(key, default))
            if isinstance(d, dict):
                return int(d.get(key, default))
        except Exception:
            pass
        return default

    def _encode_weather_block(self, battle: AbstractBattle) -> np.ndarray:
        # Weather: 4 * 9 onehots + 1 bit for "no weather" => 37 dims
        def encode_cond(active_turns: Optional[int], permanent: bool = False) -> np.ndarray:
            if permanent:
                return self._one_hot(8, 9)
            if active_turns and active_turns > 0:
                return self._one_hot(min(active_turns - 1, 7), 9)
            return np.zeros(9, dtype=np.float32)

        weather_map = getattr(battle, "weather", None)

        sun_turns = 0
        rain_turns = 0
        hail_turns = 0
        sand_turns = 0
        try:
            if weather_map:
                sun_turns = self._get_dict_value(weather_map, getattr(Weather, "SUNNYDAY", "sunnyday"), 0)
                rain_turns = self._get_dict_value(weather_map, getattr(Weather, "RAINDANCE", "raindance"), 0)
                hail_turns = max(
                    self._get_dict_value(weather_map, getattr(Weather, "HAIL", "hail"), 0),
                    self._get_dict_value(weather_map, getattr(Weather, "SNOW", "snow"), 0),
                )
                sand_turns = self._get_dict_value(weather_map, getattr(Weather, "SANDSTORM", "sandstorm"), 0)
        except Exception:
            pass

        # Permanent flags (mostly Gen 3-5); set false by default in Gen 9
        sun_perm = False
        rain_perm = False
        hail_perm = False
        sand_perm = False

        parts = [
            encode_cond(sun_turns, sun_perm),
            encode_cond(rain_turns, rain_perm),
            encode_cond(hail_turns, hail_perm),
            encode_cond(sand_turns, sand_perm),
        ]
        no_weather = 1.0 if (sun_turns + rain_turns + hail_turns + sand_turns) == 0 else 0.0
        return np.concatenate([*parts, np.array([no_weather], dtype=np.float32)])

    def _encode_trick_room(self, battle: AbstractBattle) -> np.ndarray:
        fields = getattr(battle, "fields", {}) or {}
        turns = 0
        try:
            turns = self._get_dict_value(fields, getattr(Field, "TRICK_ROOM", "trickroom"), 0)
        except Exception:
            pass
        # 7 bins: 0..5 for 1..6 turns, 6 for none
        if turns <= 0:
            return self._one_hot(6, 7)
        return self._one_hot(min(turns - 1, 5), 7)

    def _encode_force_switch(self, battle: AbstractBattle) -> np.ndarray:
        us_forced = 1.0 if getattr(battle, "must_switch", False) else 0.0
        opp_forced = 0.0  # Not directly exposed; default 0
        return np.array([us_forced, opp_forced], dtype=np.float32)

    def _encode_hazards_and_screens(self, battle: AbstractBattle) -> np.ndarray:
        # Per side: SR (2), Spikes (4), TSpikes (3), Reflect (10), Light Screen (10), Safeguard (7) => 36
        # Both sides => 72
        def get_side_dict(ours: bool) -> dict:
            key = "side_conditions" if ours else "opponent_side_conditions"
            return getattr(battle, key, {}) or {}

        def spikes_onehot(levels: int) -> np.ndarray:
            return self._one_hot(int(np.clip(levels, 0, 3)), 4)

        def tspikes_onehot(levels: int) -> np.ndarray:
            return self._one_hot(int(np.clip(levels, 0, 2)), 3)

        def duration_onehot(turns: int, length: int) -> np.ndarray:
            if turns <= 0:
                return self._one_hot(length - 1, length)  # last is "none"
            return self._one_hot(int(np.clip(turns - 1, 0, length - 2)), length)

        parts: List[np.ndarray] = []
        for ours in (True, False):
            sd = get_side_dict(ours)
            # Stealth Rock
            try:
                sr_present = getattr(SideCondition, "STEALTH_ROCK", "stealthrock") in sd
            except Exception:
                sr_present = ("stealthrock" in sd) or self._get_dict_value(sd, "stealthrock", 0) > 0
            parts.append(self._presence_one_hot(bool(sr_present)))

            # Spikes
            spikes_key = getattr(SideCondition, "SPIKES", "spikes")
            parts.append(spikes_onehot(self._get_dict_value(sd, spikes_key, 0)))

            # Toxic Spikes
            tsp_key = getattr(SideCondition, "TOXIC_SPIKES", "toxicspikes")
            parts.append(tspikes_onehot(self._get_dict_value(sd, tsp_key, 0)))

            # Reflect
            ref_key = getattr(SideCondition, "REFLECT", "reflect")
            parts.append(duration_onehot(self._get_dict_value(sd, ref_key, 0), 10))

            # Light Screen
            ls_key = getattr(SideCondition, "LIGHT_SCREEN", "lightscreen")
            parts.append(duration_onehot(self._get_dict_value(sd, ls_key, 0), 10))

            # Safeguard
            sg_key = getattr(SideCondition, "SAFEGUARD", "safeguard")
            parts.append(duration_onehot(self._get_dict_value(sd, sg_key, 0), 7))

        return np.concatenate(parts)

    # ---- Table A.2 per-Pokémon encoder (300 dims) ----

    def _idx_or_zero(self, name: Optional[str], mapping: Dict[str, int], clip_max: Optional[int] = None) -> int:
        if not name:
            return 0
        key = str(name).lower()
        idx = mapping.get(key, 0)
        if clip_max is not None:
            idx = int(np.clip(idx, 0, clip_max))
        return int(idx)

    def _hp_fraction(self, mon) -> float:
        try:
            return float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _get_boost(self, mon, key: str) -> int:
        try:
            boosts = getattr(mon, "boosts", {}) or {}
            # common aliases
            if key not in boosts and key == "accuracy":
                key = "acc"
            return int(boosts.get(key, 0))
        except Exception:
            return 0

    def _one_hot_boost(self, val: int) -> np.ndarray:
        # onehot from -6..+6 (13 bins); clamp
        v = int(np.clip(val, -6, 6))
        return self._one_hot(v + 6, 13)

    def _one_hot_duration_or_none(self, turns: int, length: int) -> np.ndarray:
        # bins 0..(length-2) => turns 1..(length-1), bin (length-1) => none
        if turns <= 0:
            return self._one_hot(length - 1, length)
        return self._one_hot(int(np.clip(turns - 1, 0, length - 2)), length)

    def _pp_bin_value(self, move) -> float:
        # Encode as {0, 0.25, 0.5, 0.75} using floor(cuberoot(pp-1))/4
        try:
            pp = int(getattr(move, "current_pp", 0) or 0)
        except Exception:
            pp = 0
        b = int(np.floor(np.cbrt(max(pp - 1, 0))))
        b = int(np.clip(b, 0, 3))
        return float(b) / 4.0

    def _last_used_move_index(self, mon) -> int:
        # Try common attributes
        for attr in ("last_move", "last_used_move", "last_used_move_name"):
            mv = getattr(mon, attr, None)
            if mv:
                try:
                    name = getattr(mv, "id", None) or getattr(mv, "name", None) or str(mv)
                    return self._idx_or_zero(name, self._move_index, 198)
                except Exception:
                    pass
        return 0

    def _get_effects_dict(self, mon) -> Dict[str, Any]:
        # Try a few attribute names that poke-env might use for volatile effects
        for attr in ("effects", "volatile_statuses", "volatile", "volatiles", "statuses"):
            eff = getattr(mon, attr, None)
            if isinstance(eff, dict):
                return eff
            if isinstance(eff, set):
                # convert set to dict of True
                return {str(x).lower(): True for x in eff}
        return {}

    def _has_effect(self, effs: Dict[str, Any], name: str) -> bool:
        if not effs:
            return False
        low = name.lower()
        if low in effs:
            return True
        # some effects may be objects with .name or .id
        for k, v in effs.items():
            try:
                if isinstance(k, str) and k.lower() == low:
                    return True
                nm = getattr(v, "name", None) or getattr(v, "id", None)
                if nm and str(nm).lower() == low:
                    return True
            except Exception:
                pass
        return False

    def _effect_turns(self, effs: Dict[str, Any], name: str) -> int:
        # Try to extract a "turns" or "duration" int from the effect entry
        if not effs:
            return 0
        low = name.lower()
        entry = None
        if low in effs:
            entry = effs[low]
        else:
            for k, v in effs.items():
                if isinstance(k, str) and k.lower() == low:
                    entry = v
                    break
        if entry is None:
            return 0
        try:
            for key in ("turns", "duration", "counter", "remaining"):
                if hasattr(entry, key):
                    return int(getattr(entry, key) or 0)
                if isinstance(entry, dict) and key in entry:
                    return int(entry[key] or 0)
        except Exception:
            pass
        # some effects store remaining as the int itself
        try:
            return int(entry)
        except Exception:
            return 0

    def _type_one_hot(self, mon) -> np.ndarray:
        vec = np.zeros(18, dtype=np.float32)
        try:
            types = getattr(mon, "types", None)
            if types:
                for t in types:
                    # t can be enum-like; use name or str
                    name = (getattr(t, "name", None) or str(t)).lower()
                    idx = self._type_to_idx.get(name, None)
                    if idx is not None and 0 <= idx < 18:
                        vec[idx] = 1.0
        except Exception:
            pass
        return vec

    def _gender_one_hot(self, mon) -> np.ndarray:
        # male, female, neutral (3)
        g = (getattr(mon, "gender", None) or "").lower()
        if g.startswith("m"):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if g.startswith("f"):
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _status_one_hot(self, mon) -> np.ndarray:
        # 7 bins: BRN, PAR, PSN, TOX, FRZ, SLP, FNT
        order = ["brn", "par", "psn", "tox", "frz", "slp"]
        vec = np.zeros(7, dtype=np.float32)
        hp0 = self._hp_fraction(mon) <= 0.0
        if hp0:
            vec[6] = 1.0
            return vec
        st = getattr(mon, "status", None)
        if st:
            s = str(st).lower()
            for i, name in enumerate(order):
                if name in s:
                    vec[i] = 1.0
                    break
        return vec

    def _toxic_counter_one_hot(self, mon) -> np.ndarray:
        # 21 bins for # turns (0..20)
        turns = 0
        try:
            st = getattr(mon, "status", None)
            if st and "tox" in str(st).lower():
                turns = int(getattr(mon, "status_counter", 0) or 0)
        except Exception:
            turns = 0
        turns = int(np.clip(turns, 0, 20))
        return self._one_hot(turns, 21)

    def _sleep_counter_one_hot(self, mon) -> np.ndarray:
        # 11 bins for # turns (0..10)
        turns = 0
        try:
            st = getattr(mon, "status", None)
            if st and "slp" in str(st).lower():
                turns = int(getattr(mon, "status_counter", 0) or 0)
        except Exception:
            turns = 0
        turns = int(np.clip(turns, 0, 10))
        return self._one_hot(turns, 11)

    def _log10_one_hot(self, value: Optional[float], bins: int, clamp: Tuple[int, int]) -> np.ndarray:
        # Round log10(value) and map to [clamp[0], clamp[1]] inclusive, then shift to 0..bins-1
        vec = np.zeros(bins, dtype=np.float32)
        if value is None or value <= 0:
            vec[0] = 1.0
            return vec
        lg = int(np.round(np.log10(max(value, 1e-8))))
        lg = int(np.clip(lg, clamp[0], clamp[1]))
        idx = lg - clamp[0]  # shift to start at 0
        idx = int(np.clip(idx, 0, bins - 1))
        vec[idx] = 1.0
        return vec

    def _is_preparing(self, effs: Dict[str, Any]) -> bool:
        for nm in ("fly", "bounce", "dig", "dive", "shadowforce", "phantomforce", "skydrop", "skyattack", "solarbeam", "solarblade", "geomancy"):
            if self._has_effect(effs, nm):
                return True
        return False

    def _pokemon_weight_height(self, species_name: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
        if not species_name:
            return None, None
        key = str(species_name).lower()
        # poke-env stores pokedex data under gen_data.pokedex, keys lowercase species names
        try:
            dex = getattr(self._gen_data, "pokedex", None) or {}
            data = dex.get(key, None)
            if isinstance(data, dict):
                w = data.get("weight_kg", None)
                h = data.get("height_m", None)
                return (float(w) if w is not None else None, float(h) if h is not None else None)
        except Exception:
            pass
        return None, None

    def _encode_one_pokemon(self, mon, is_opponent: bool, is_active: bool) -> np.ndarray:
        # Unknown slot sentinel: if None or no species known
        if mon is None:
            vec = np.zeros(300, dtype=np.float32)
            vec[-1] = 1.0  # unknown bit
            # mark is_opponent and active even if unknown (keeps slot identity)
            vec[-3] = 1.0 if is_active else 0.0         # active onehot idx 1 at -3/-2 region
            vec[-2] = 1.0 if is_opponent else 0.0       # is_opponent onehot idx 1 at -2/-1 region
            return vec

        species_name = getattr(mon, "species", None)
        unknown = species_name is None or str(species_name).lower() in ("unknown", "unrevealed", "")

        # Precompute effects
        effs = self._get_effects_dict(mon)

        # Scalars (clipped indices per table domains)
        species_idx = self._idx_or_zero(species_name, self._species_index, 295)
        ability_idx = self._idx_or_zero(getattr(mon, "ability", None), self._ability_index, 100)
        item_idx = self._idx_or_zero(getattr(mon, "item", None), self._item_index, 39)

        # Moves: 4 ids + 4 PP bin scalars
        move_ids: List[int] = [0, 0, 0, 0]
        move_pp_bins: List[float] = [0.0, 0.0, 0.0, 0.0]
        try:
            moves_list = list(getattr(mon, "moves", {}).values())
        except Exception:
            moves_list = []
        for i in range(min(4, len(moves_list))):
            mv = moves_list[i]
            mv_name = getattr(mv, "id", None) or getattr(mv, "name", None)
            move_ids[i] = self._idx_or_zero(mv_name, self._move_index, 198)
            move_pp_bins[i] = self._pp_bin_value(mv)

        last_move_idx = self._last_used_move_index(mon)

        # Types
        types_vec = self._type_one_hot(mon)

        # HP fraction binning
        hp_bin = self._bin_hp_fraction(self._hp_fraction(mon))

        # Boosts
        boosts = [
            self._one_hot_boost(self._get_boost(mon, "accuracy")),
            self._one_hot_boost(self._get_boost(mon, "atk")),
            self._one_hot_boost(self._get_boost(mon, "def")),
            self._one_hot_boost(self._get_boost(mon, "evasion")),
            self._one_hot_boost(self._get_boost(mon, "spa")),
            self._one_hot_boost(self._get_boost(mon, "spd")),
            self._one_hot_boost(self._get_boost(mon, "spe")),
        ]

        # Volatile effects (38 effects × 2 onehot [OFF/ON])
        vol_parts: List[np.ndarray] = []
        for name in self._volatile_names:
            on = self._has_effect(effs, name)
            vol_parts.append(self._presence_one_hot(on))
        vol_vec = np.concatenate(vol_parts)

        # Encore, Taunt, Magnet Rise, Slow Start durations (or none)
        encore_vec = self._one_hot_duration_or_none(self._effect_turns(effs, "encore"), 9)
        taunt_vec = self._one_hot_duration_or_none(self._effect_turns(effs, "taunt"), 6)
        mrise_vec = self._one_hot_duration_or_none(self._effect_turns(effs, "magnet rise"), 7)
        sslow_vec = self._one_hot_duration_or_none(self._effect_turns(effs, "slowstart"), 6)

        # Gender (3)
        gender_vec = self._gender_one_hot(mon)

        # Status (7), Toxic and Sleep counters
        status_vec = self._status_one_hot(mon)
        tox_vec = self._toxic_counter_one_hot(mon)
        slp_vec = self._sleep_counter_one_hot(mon)

        # Weight and Height onehots: log10(weight) => 5 bins (clamp [-2..2]), log10(height) => 4 bins (clamp [-1..2])
        w_kg, h_m = self._pokemon_weight_height(species_name if not unknown else None)
        weight_vec = self._log10_one_hot(w_kg, bins=5, clamp=(-2, 2))
        height_vec = self._log10_one_hot(h_m, bins=4, clamp=(-1, 2))

        # First turn, Protect counter, Must recharge, Preparing, Active, Is opponent
        first_turn = False
        if hasattr(mon, "just_switched"):
            first_turn = bool(getattr(mon, "just_switched"))
        elif hasattr(mon, "first_turn"):
            first_turn = bool(getattr(mon, "first_turn"))
        first_turn_vec = self._presence_one_hot(first_turn)

        # Protect counter (0..5) -> 6 onehot (no "none" separate)
        prot_count = 0
        for nm in ("protectcounter", "protect_counter", "protects_in_row", "consecutive_protects"):
            if hasattr(mon, nm):
                try:
                    prot_count = int(getattr(mon, nm) or 0)
                    break
                except Exception:
                    pass
        prot_count = int(np.clip(prot_count, 0, 5))
        protect_vec = self._one_hot(prot_count, 6)

        # Must recharge: volatile or boolean attribute
        must_recharge = self._has_effect(effs, "mustrecharge") or bool(getattr(mon, "must_recharge", False))
        must_recharge_vec = self._presence_one_hot(must_recharge)

        # Preparing two-turn moves
        preparing = self._is_preparing(effs)
        preparing_vec = self._presence_one_hot(preparing)

        active_vec = self._presence_one_hot(is_active)
        opp_vec = self._presence_one_hot(is_opponent)

        # Unknown flag: if true, zero all others and set unknown bit
        if unknown:
            vec = np.zeros(300, dtype=np.float32)
            # keep active/opponent flags even if unknown
            vec[-3] = 1.0 if is_active else 0.0
            vec[-2] = 1.0 if is_opponent else 0.0
            vec[-1] = 1.0
            return vec

        # Assemble vector in the exact Table A.2 order (length sums to 300)
        parts: List[np.ndarray] = []
        parts.append(np.array([float(species_idx)], dtype=np.float32))  # species (1)
        parts.append(np.array([float(ability_idx)], dtype=np.float32))  # ability (1)
        parts.append(np.array([float(item_idx)], dtype=np.float32))     # item (1)
        parts.append(np.array([float(x) for x in move_ids], dtype=np.float32))  # move (4)
        parts.append(np.array([float(x) for x in move_pp_bins], dtype=np.float32))  # ⌊∛PP⌋/4 (4)
        parts.append(np.array([float(last_move_idx)], dtype=np.float32))  # last used move (1)
        parts.append(types_vec.astype(np.float32))  # types (18)
        parts.append(hp_bin.astype(np.float32))     # current hp fraction (7)
        # boosts
        parts.append(boosts[0])  # accuracy (13)
        parts.append(boosts[1])  # atk (13)
        parts.append(boosts[2])  # def (13)
        parts.append(boosts[3])  # evasion (13)
        parts.append(boosts[4])  # spa (13)
        parts.append(boosts[5])  # spd (13)
        parts.append(boosts[6])  # spe (13)
        # volatile effects 2*38 (76)
        parts.append(vol_vec.astype(np.float32))
        # durations
        parts.append(encore_vec.astype(np.float32))  # 9
        parts.append(taunt_vec.astype(np.float32))   # 6
        parts.append(mrise_vec.astype(np.float32))   # 7
        parts.append(sslow_vec.astype(np.float32))   # 6
        # misc
        parts.append(gender_vec.astype(np.float32))  # 3
        parts.append(status_vec.astype(np.float32))  # 7
        parts.append(tox_vec.astype(np.float32))     # 21
        parts.append(slp_vec.astype(np.float32))     # 11
        parts.append(weight_vec.astype(np.float32))  # 5
        parts.append(height_vec.astype(np.float32))  # 4
        parts.append(first_turn_vec.astype(np.float32))  # 2
        parts.append(protect_vec.astype(np.float32))     # 6
        parts.append(must_recharge_vec.astype(np.float32))  # 2
        parts.append(preparing_vec.astype(np.float32))      # 2
        parts.append(active_vec.astype(np.float32))         # 2
        parts.append(opp_vec.astype(np.float32))            # 2
        # unknown (1)
        parts.append(np.array([0.0], dtype=np.float32))

        vec = np.concatenate(parts).astype(np.float32)
        if vec.shape[0] != 300:
            raise RuntimeError(f"Per-Pokémon vector size mismatch: {vec.shape[0]} != 300")
        return vec

    def _ordered_team(self, battle: AbstractBattle) -> Tuple[List[Any], List[Any]]:
        # Ensure 6 slots for each side; place active first, then others, pad with None as unknowns
        ours: List[Any] = []
        opps: List[Any] = []
        active = getattr(battle, "active_pokemon", None)
        if active:
            ours.append(active)
        ours_rest = [m for m in getattr(battle, "team", {}).values() if m is not active]
        ours.extend(ours_rest)
        ours = (ours + [None] * 6)[:6]

        opp_active = getattr(battle, "opponent_active_pokemon", None)
        if opp_active:
            opps.append(opp_active)
        opps_rest = [m for m in getattr(battle, "opponent_team", {}).values() if m is not opp_active]
        opps.extend(opps_rest)
        opps = (opps + [None] * 6)[:6]

        return ours, opps

    def _encode_pokemon_block(self, battle: AbstractBattle) -> np.ndarray:
        ours, opps = self._ordered_team(battle)
        vecs: List[np.ndarray] = []
        for i, mon in enumerate(ours):
            vecs.append(self._encode_one_pokemon(mon, is_opponent=False, is_active=(i == 0)))
        for i, mon in enumerate(opps):
            vecs.append(self._encode_one_pokemon(mon, is_opponent=True, is_active=(i == 0)))
        block = np.concatenate(vecs).astype(np.float32)
        if block.shape[0] != 12 * 300:
            raise RuntimeError(f"Pokémon block size mismatch: {block.shape[0]} != {12*300}")
        return block

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        3725-dim vector per Appendix A:
        - Weather (9*4 + 1 = 37)
        - Trick Room (7)
        - Force Switch (2)
        - Unknown (7)  [reserved extra effects block]
        - Hazards & Screens (SR 2*2, Spikes 2*4, TSpikes 2*3, Reflect 2*10, Light Screen 2*10, Safeguard 2*7) = 72
        - Pokémon: 12 * 300 = 3600
        Total = 37 + 7 + 2 + 7 + 72 + 3600 = 3725
        """
        weather = self._encode_weather_block(battle)       # 37
        trick = self._encode_trick_room(battle)            # 7
        forced = self._encode_force_switch(battle)         # 2
        unknown = np.zeros(7, dtype=np.float32)            # placeholder per Table A.1
        hazards = self._encode_hazards_and_screens(battle) # 72
        pokes = self._encode_pokemon_block(battle)         # 3600

        final_vec = np.concatenate([weather, trick, forced, unknown, hazards, pokes]).astype(np.float32)
        if final_vec.shape[0] != self._observation_size():
            raise RuntimeError(f"Observation size mismatch: got {final_vec.shape[0]}, expected {self._observation_size()}")
        return final_vec


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