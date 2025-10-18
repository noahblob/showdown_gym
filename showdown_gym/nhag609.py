from poke_env.battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from poke_env.player import Player
from poke_env.battle.pokemon_type import PokemonType

team = """
Glimmora @ Focus Sash  
Ability: Toxic Debris  
Tera Type: Rock  
EVs: 252 HP / 252 Def  
Lax Nature  
- Spikes    
- Mortal Spin  
- Sludge Wave  
- Power Gem 

Iron Moth @ Booster Energy  
Ability: Quark Drive  
Tera Type: Fire  
EVs: 124 HP / 132 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Discharge  
- Flamethrower  
- Sludge Wave  
- Energy Ball  

Koraidon @ Expert Belt  
Ability: Orichalcum Pulse  
Tera Type: Fighting  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Collision Course  
- Dragon Claw  
- Flare Blitz  
- Iron Head  

Flutter Mane @ Life Orb  
Ability: Protosynthesis  
Tera Type: Fire  
EVs: 4 Def / 252 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Moonblast  
- Shadow Ball  
- Mystical Fire  
- Power Gem  

Zacian-Crowned @ Rusted Sword  
Ability: Intrepid Sword  
Tera Type: Fairy  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
- Sacred Sword  
- Behemoth Blade  
- Play Rough  
- Ice Fang  

Deoxys-Attack @ Focus Sash  
Ability: Pressure  
Tera Type: Ice  
EVs: 4 HP / 252 SpA / 252 Spe  
Hasty Nature  
IVs: 0 Atk  
- Psychic
- Ice Beam  
- Focus Blast  
- Thunderbolt
"""


class CustomAgent(Player):
    # hazard constants
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealthrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(team=team, *args, **kwargs)

        # Keep track of turns and switches
        self.turns_this_battle = 0.0
        self.switches_this_battle = 0.0
        self.good_switches_this_battle = 0.0
        self.bad_switches_this_battle = 0.0
        self.turns_history = []
        self.switches_history = []
        self.good_switches_history = []
        self.bad_switches_history = []

        # State tracking for opponent switches after KO
        self.last_opponent_species = None
        self.opponent_switch_after_ko = False
        self.last_opponent_hp = None

    def teampreview(self, battle: AbstractBattle) -> str:
        # Lead with the first Pokemon, keep default order for the rest
        return "/team 123456"

    def _is_move_immune(self, move_type, opp_ability):
        """Check if move is immune due to opponent's ability"""
        if not opp_ability:
            return False

        ability_name = opp_ability.lower()
        immunity_map = {
            "levitate": PokemonType.GROUND,
            "flashfire": PokemonType.FIRE,
            "waterabsorb": PokemonType.WATER,
            "dryskin": PokemonType.WATER,
            "sapsipper": PokemonType.GRASS,
            "lightningrod": PokemonType.ELECTRIC,
            "stormdrain": PokemonType.WATER,
        }

        return immunity_map.get(ability_name) == move_type

    def _estimate_effectiveness(self, move_type, opponent, my_types):
        """Estimate move effectiveness, boosting if move matches user's type"""
        eff = opponent.damage_multiplier(move_type)
        if move_type in my_types:
            eff *= 1.5
        return eff

    def _estimate_damage_frac(self, mv, my_types, opponent):
        """Estimate damage as a fraction of full HP (0.0 to 1.0)"""
        bp = mv.base_power if mv.base_power else 0
        if bp <= 0:
            return 0.0

        move_type = mv.type if mv.type else mv.type_id
        if move_type is None:
            return 0.0

        eff = self._estimate_effectiveness(move_type, opponent, my_types)
        return max(0.0, min((bp / 100) * eff, 1.0))

    def _opponent_has_advantage(self, me, opp_types, thresh=1.0):
        """Check if opponent has type advantage over me above a threshold"""
        return any(me.damage_multiplier(otype) > thresh for otype in opp_types)

    def _pick_best_switch(self, battle, opponent):
        """Pick the best switch-in based on type matchups."""
        best_mon, best_score = None, float("-inf")
        for mon in battle.available_switches:
            # Defensive risk: how hard opponent's typing hits this mon
            worst_incoming = max(
                mon.damage_multiplier(otype) for otype in opponent.types
            )

            # Offensive advantage: best of this mon's types vs opponent's types
            best_offense = 0.0
            if mon.types:
                best_offense = max(
                    opponent.damage_multiplier(ct) for ct in mon.types if ct is not None
                )

            # Prioritise type advantage (effective) and defensive safety
            score = (best_offense * 2.0) / (worst_incoming + 0.5)

            if score > best_score:
                best_score, best_mon = score, mon

        if best_mon:
            pass
        return best_mon

    def _pick_best_switch_advanced(
        self,
        battle,
        me,
        opp,
        opp_types,
        effective_moves,
        active_defensive_risk,
        current_score,
    ):
        """Determine best switch-in based on multiple factors."""
        best_switch = None
        best_score = current_score
        for candidate in battle.available_switches:
            # Offensive advantage: best of candidate's types vs opponent's types
            offensive_advantage = 0.0
            if candidate.types:
                offensive_advantage = max(
                    opp.damage_multiplier(ct)
                    for ct in candidate.types
                    if ct is not None
                )

            # Only consider candidates with clear offensive advantage (>1x)
            if offensive_advantage > 1.0:
                # Defensive risk: how hard opp's typing hits candidate's typing
                defensive_risk = 1.0
                if candidate.types:
                    defensive_risk = max(
                        candidate.damage_multiplier(ot)
                        for ot in opp_types
                        if ot is not None
                    )

                # More leniency on defensive risk for switch-in if we already have type advantage
                if defensive_risk >= 2.0 and active_defensive_risk < 2.0:
                    continue

                # Calculate score with multiple factors for better decision making
                # Base score from offensive advantage
                base_score = offensive_advantage * 2.0 - defensive_risk

                # Bonus for better defensive typing
                defensive_bonus = 0.0
                if defensive_risk < active_defensive_risk:
                    defensive_bonus = (active_defensive_risk - defensive_risk) * 0.5

                # Bonus for being the current active Pokemon
                current_bonus = 0.0
                if candidate.species == me.species:
                    current_bonus = 1.0

                # Penalty for switching when we have effective moves
                switch_penalty = 0.0
                if effective_moves:
                    switch_penalty = 1.0

                score = base_score + defensive_bonus + current_bonus - switch_penalty

                if score > best_score:
                    best_score = score
                    best_switch = candidate

        if best_switch is not None and best_switch.species != me.species:
            return best_switch
        return None

    def _check_side_conditions(self, battle):
        """Check if there are any hazards on our side."""
        if battle.side_conditions:
            for condition in battle.side_conditions:
                if condition in self.ENTRY_HAZARDS.values():
                    return True
        return False

    def _check_enemy_hazards(self, battle):
        """Check if there are any hazards on opponent side."""
        if battle.opponent_side_conditions:
            for condition in battle.opponent_side_conditions:
                if condition in self.ENTRY_HAZARDS.values():
                    return True
        return False

    def _find_effective_moves(self, battle, pokemon, opponent):
        """
        Calculates different categories of moves based on effectiveness.
        Returns a dict with keys:
        'x6', 'x4', 'x3', 'x2', 'stab', 'neutral', 'resisted', 'immune'
        Each value is a list of (move, effectiveness, damage_estimate).
        """
        categories = {
            "x6": [],
            "x4": [],
            "x3": [],
            "x2": [],
            "stab": [],
            "neutral": [],
            "resisted": [],
            "immune": [],
        }
        my_types = pokemon.types

        weather = None
        terrain = None
        if hasattr(battle, "_weather"):
            for field in battle._weather:
                # Weather
                if hasattr(field, "name") and "weather" in field.name.lower():
                    weather = field.name.lower()
                # Terrain
                if hasattr(field, "name") and "terrain" in field.name.lower():
                    terrain = field.name.lower()
                # If using enum, check type
                if "Weather" in str(type(field)):
                    weather = field.name.lower()
                if "Terrain" in str(type(field)):
                    terrain = field.name.lower()

        if isinstance(getattr(battle, "_weather", None), dict):
            for k in battle._weather:
                # Weather
                if "Weather" in str(type(k)):
                    weather = k.name.lower()
                # Terrain
                if "Terrain" in str(type(k)):
                    terrain = k.name.lower()

        def weather_boost(move_type):
            if weather == "sunnyday":
                if move_type == PokemonType.FIRE:
                    return 1.5
                if move_type == PokemonType.WATER:
                    return 0.5
            elif weather == "raindance":
                if move_type == PokemonType.WATER:
                    return 1.5
                if move_type == PokemonType.FIRE:
                    return 0.5
            return 1.0

        def terrain_boost(move_type):
            if terrain == "electricterrain" and move_type == PokemonType.ELECTRIC:
                return 1.3
            if terrain == "grassyterrain" and move_type == PokemonType.GRASS:
                return 1.3
            if terrain == "psychicterrain" and move_type == PokemonType.PSYCHIC:
                return 1.3
            if terrain == "mistyterrain" and move_type == PokemonType.DRAGON:
                return 0.5
            return 1.0

        for mv in battle.available_moves:
            move_type = mv.type if mv.type else mv.type_id
            if move_type is None:
                continue

            # Ability-based immunities
            opp_ability = opponent.ability if opponent else None
            if self._is_move_immune(move_type, opp_ability):
                categories["immune"].append((mv, 0.0, 0.0))
                continue

            # Type effectiveness
            type_mult = opponent.damage_multiplier(move_type)

            # same type attack bonus (STAB)
            stab = 1.5 if move_type in my_types else 1.0

            # Weather/terrain boost
            wboost = weather_boost(move_type)
            tboost = terrain_boost(move_type)

            # Total effectiveness
            effectiveness = type_mult * stab * wboost * tboost

            # Estimate damage
            dmg = self._estimate_damage_frac(mv, my_types, opponent)

            # Categorise
            if effectiveness >= 6.0:
                categories["x6"].append((mv, dmg, effectiveness))
            elif effectiveness >= 4.0:
                categories["x4"].append((mv, dmg, effectiveness))
            elif effectiveness >= 3.0:
                categories["x3"].append((mv, dmg, effectiveness))
            elif effectiveness >= 2.0:
                categories["x2"].append((mv, dmg, effectiveness))
            elif stab > 1.0 and type_mult == 1.0:
                categories["stab"].append((mv, dmg, effectiveness))
            elif abs(effectiveness - 1.0) < 1e-6:
                categories["neutral"].append((mv, dmg, effectiveness))
            elif 0.0 < effectiveness < 1.0:
                categories["resisted"].append((mv, dmg, effectiveness))
            elif effectiveness == 0.0:
                categories["immune"].append((mv, dmg, effectiveness))

        return categories

    def _get_best_effective_move(self, effective_moves, thresh=1.0):
        """
        Gets the best effective move with minimum threshold effectiveness.
        Prioritises by effectiveness first, then by estimated damage.
        effective_moves: list of (move, damage_estimate, effectiveness)
        Returns: move or None
        """
        filtered = [m for m in effective_moves if m[2] >= thresh]
        if not filtered:
            return None
        # Sort by effectiveness (descending), then damage (descending)
        filtered.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return filtered[0][0]

    def choose_move(self, battle: AbstractBattle):
        # Throw exception if opponent has no active Pokemon (should not happen in normal play)
        if battle.opponent_active_pokemon is None:
            raise ValueError("Opponent has no active Pokemon!")
        
        # --- Track turns ---
        self.turns_this_battle += 1
        # --- END track turns ---

        # Basic info
        opp = battle.opponent_active_pokemon
        me = battle.active_pokemon
        opp_types = opp.types
        opp_hp = opp.current_hp_fraction or 1.0

        current_opp_species = opp.species if opp else None
        current_opp_hp = opp_hp

        # Check if opponent just switched after we KO'd their last Pokemon
        if (
            self.last_opponent_species is not None
            and current_opp_species != self.last_opponent_species
            and self.last_opponent_hp is not None
            and self.last_opponent_hp <= 0.0
        ):
            self.opponent_switch_after_ko = True
        else:
            self.opponent_switch_after_ko = False

        self.last_opponent_species = current_opp_species
        self.last_opponent_hp = current_opp_hp

        # Categorise moves
        move_categories = self._find_effective_moves(battle, me, opp)

        effective_moves = (
            move_categories["x6"]
            + move_categories["x4"]
            + move_categories["x3"]
            + move_categories["x2"]
        )

        moves = (
            effective_moves
            + move_categories["stab"]
            + move_categories["neutral"]
            + move_categories["resisted"]
        )

        # Defensive risk of staying in: how hard opp's typing hits our current active typing
        active_defensive_risk = max(
            (me.damage_multiplier(ot) for ot in opp_types if ot is not None),
            default=1.0,
        )

        # Forced to switch?
        if not battle.available_moves and battle.available_switches:
            self.switches_this_battle += 1
            counter = self._pick_best_switch(battle, opp)
            if counter:

                # --- Record good/bad switch based on offensive/defensive matchup ---
                curr_off = 0.0
                if me.types:
                    curr_off = max(
                        opp.damage_multiplier(ct) for ct in me.types if ct is not None
                    )
                curr_def = max(
                    (me.damage_multiplier(ot) for ot in opp_types if ot is not None),
                    default=1.0,
                )
                counter_off = 0.0
                if counter.types:
                    counter_off = max(
                        opp.damage_multiplier(ct)
                        for ct in counter.types
                        if ct is not None
                    )
                counter_def = max(
                    (
                        counter.damage_multiplier(ot)
                        for ot in opp_types
                        if ot is not None
                    ),
                    default=1.0,
                )
                if (counter_off > curr_off) or (counter_def < curr_def):
                    self.good_switches_this_battle += 1
                else:
                    self.bad_switches_this_battle += 1
                # --- END record good/bad switch ---

                return self.create_order(counter)
            # If no counter, just switch (record as bad)
            self.bad_switches_this_battle += 1
            return self.create_order(list(battle.available_switches)[0])

        # If opponent just switched after we KO'd, prioritise super effective moves
        if self.opponent_switch_after_ko:
            best_move = self._get_best_effective_move(effective_moves, thresh=2.0)
            if best_move:
                return self.create_order(best_move)

        # If we're faster, check if we can KO first
        if me.base_stats["spe"] > opp.base_stats["spe"]:
            # First, check if any super effective moves can KO
            for m, dmg, mult in sorted(
                effective_moves,
                key=lambda x: (x[2], x[1]),
                reverse=True,
            ):
                if dmg >= current_opp_hp:
                    return self.create_order(m)

            # If no super effective moves can KO, then check other moves
            for m, dmg, mult in sorted(moves, key=lambda x: x[1], reverse=True):
                if dmg >= current_opp_hp and mult >= 1.0:
                    return self.create_order(m)

        # --- Hazard management ---
        enemy_hazards = self._check_enemy_hazards(battle)

        if not enemy_hazards:
            for mv in battle.available_moves:
                if mv.id in ["spikes", "stealthrock", "toxicspikes", "stickyweb"]:
                    return self.create_order(mv)

        hazards_detected = self._check_side_conditions(battle)

        if hazards_detected:
            for mv in battle.available_moves:
                if mv.id in ["mortalspin", "defog"]:
                    return self.create_order(mv)
        # --- END hazard management ---

        # Cannot KO? Try to switch if we have a type advantage
        current_offensive_advantage = 0.0
        if me.types:
            current_offensive_advantage = max(
                opp.damage_multiplier(ct) for ct in me.types if ct is not None
            )
        current_base_score = current_offensive_advantage * 2.0 - active_defensive_risk
        current_bonus = 1.0
        switch_penalty = 1.0 if effective_moves else 0.0
        current_score = current_base_score + current_bonus - switch_penalty

        best_switch = None
        best_score = current_score

        if battle.available_switches:
            best_switch = self._pick_best_switch_advanced(
                battle,
                me,
                opp,
                opp_types,
                effective_moves,
                active_defensive_risk,
                best_score,
            )
            if best_switch is not None:

                # --- Record good/bad switch based on offensive/defensive matchup ---
                self.switches_this_battle += 1
                curr_off = 0.0
                if me.types:
                    curr_off = max(
                        opp.damage_multiplier(ct) for ct in me.types if ct is not None
                    )
                curr_def = max(
                    (me.damage_multiplier(ot) for ot in opp_types if ot is not None),
                    default=1.0,
                )
                switch_off = 0.0
                if best_switch.types:
                    switch_off = max(
                        opp.damage_multiplier(ct)
                        for ct in best_switch.types
                        if ct is not None
                    )
                switch_def = max(
                    (
                        best_switch.damage_multiplier(ot)
                        for ot in opp_types
                        if ot is not None
                    ),
                    default=1.0,
                )
                if (switch_off > curr_off) or (switch_def < curr_def):
                    self.good_switches_this_battle += 1
                else:
                    self.bad_switches_this_battle += 1
                # -- - END record good/bad switch ---

                return self.create_order(best_switch)

        # Try using the most effective move left in effective moves
        if effective_moves:
            best_se_move = max(
                effective_moves, key=lambda x: (x[2], x[1])
            )  # (effectiveness, damage)
            return self.create_order(best_se_move[0])

        # Try to switch again if no effective moves, but only if counter has effective moves or we're at a disadvantage
        # and only if its defensive risk is strictly lower than our current defensive risk
        if battle.available_switches:
            counter = self._pick_best_switch(battle, opp)
            if counter:
                # Check if the counter has super effective moves against opponent
                counter_has_effective = False
                for mv in counter.moves:
                    if hasattr(mv, "type") and mv.type:
                        eff = opp.damage_multiplier(mv.type)
                        if eff > 1.0:
                            counter_has_effective = True
                            break

                disadvantaged = self._opponent_has_advantage(me, opp_types, thresh=1.0)
                counter_defensive_risk = max(
                    (
                        counter.damage_multiplier(ot)
                        for ot in opp_types
                        if ot is not None
                    ),
                    default=1.0,
                )
                if (counter_has_effective or disadvantaged) and (
                    counter_defensive_risk < active_defensive_risk
                ):

                    # --- Record good/bad switch based on offensive/defensive matchup ---
                    self.switches_this_battle += 1
                    curr_off = 0.0
                    if me.types:
                        curr_off = max(
                            opp.damage_multiplier(ct)
                            for ct in me.types
                            if ct is not None
                        )
                    curr_def = max(
                        (
                            me.damage_multiplier(ot)
                            for ot in opp_types
                            if ot is not None
                        ),
                        default=1.0,
                    )
                    counter_off = 0.0
                    if counter.types:
                        counter_off = max(
                            opp.damage_multiplier(ct)
                            for ct in counter.types
                            if ct is not None
                        )
                    if (counter_off > curr_off) or (counter_defensive_risk < curr_def):
                        self.good_switches_this_battle += 1
                    else:
                        self.bad_switches_this_battle += 1
                    # --- END record good/bad switch ---

                    return self.create_order(counter)

        # Check if we can KO with any move
        for m, dmg, mult in sorted(moves, key=lambda x: x[1], reverse=True):
            if dmg >= current_opp_hp and mult >= 1.0:
                return self.create_order(m)

        # If no effective moves, no switch-in and no KO, use neutral moves
        if move_categories["neutral"]:
            best_neutral_move = max(move_categories["neutral"], key=lambda x: x[1])
            return self.create_order(best_neutral_move[0])

        # If resisted moves only, consider switching if we have a better matchup
        # but be more conservative (just pick the best switch)
        if move_categories["resisted"] and battle.available_switches:
            counter = self._pick_best_switch(battle, opp)
            if counter:

                # --- Record good/bad switch based on offensive/defensive matchup ---
                counter_defensive_risk = max(
                    (
                        counter.damage_multiplier(ot)
                        for ot in opp_types
                        if ot is not None
                    ),
                    default=1.0,
                )
                curr_def = max(
                    (me.damage_multiplier(ot) for ot in opp_types if ot is not None),
                    default=1.0,
                )
                if counter_defensive_risk < active_defensive_risk:
                    self.switches_this_battle += 1
                    # Good switch: strictly better defensive matchup
                    if counter_defensive_risk < curr_def:
                        self.good_switches_this_battle += 1
                    else:
                        self.bad_switches_this_battle += 1
                    # --- END record good/bad switch ---

                    return self.create_order(counter)

        # If no better switch, use the highest damage resisted move
        if move_categories["resisted"]:
            best_resisted_move = max(move_categories["resisted"], key=lambda x: x[1])
            return self.create_order(best_resisted_move[0])

        # Screw it, just pick a random move
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self.turns_history.append(self.turns_this_battle)
        self.switches_history.append(self.switches_this_battle)
        self.good_switches_history.append(self.good_switches_this_battle)
        self.bad_switches_history.append(self.bad_switches_this_battle)
        self.turns_this_battle = 0
        self.switches_this_battle = 0
        self.good_switches_this_battle = 0
        self.bad_switches_this_battle = 0
