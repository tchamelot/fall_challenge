import sys
import numpy as np
from enum import IntEnum


# enum for types
class Types(IntEnum):
    BREW = 0
    CAST = 1
    OPPONENT_CAST = 2
    LEARN = 3
    REST = 4


class Col(IntEnum):
    ID = 0
    TYPE = 1
    D1 = 2
    D2 = 3
    D3 = 4
    D4 = 5
    PRICE = 6
    TOME_INDEX = 7
    TAX = 8
    CASTABLE = 9
    REPEATABLE = 10


# dict used to parse input
types = {"BREW": 0, "CAST": 1, "OPPONENT_CAST": 2, "LEARN": 3, "REST": 4}

inv_types = {0: "BREW", 1: "CAST", 2: "OPPONENT_CAST", 3: "LEARN", 4: "REST"}

# function used to parse each input
action_parsers = [
    int,
    lambda x: types[x],
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    lambda x: x != "0",
    lambda x: x != "0",
]


def score_state(state, inventory):
    """
    Args:
        state: state of the game
        inventory: inventory of the 2 players

    Returns: the score describing the "goodness" of a state

    """
    return (
        np.clip(
            state[state[:, Col.TYPE] == Types.BREW, Col.D1 : Col.D4 + 1] + inventory[:-1],
            None,
            0,
        )
        .sum(axis=1)
        .min()
        + inventory[-1]
    )


def available_actions(state, inventory):
    """
    Args:
        state: the state of the game (after removing other player actions) the inventory of the player
        inventory: the inventory of the player

    Returns: the list of available action, as a subset of state

    """
    new_inventories = state[:, Col.D1 : Col.D4 + 1] + inventory[:4]
    craftable = (new_inventories.min(axis=1) >= 0) & (state[:, Col.CASTABLE] == 1)
    is_learn = state[:, Col.TYPE] == Types.LEARN
    learnable = (
        state[:, Col.TOME_INDEX] <= inventory[0]
    )  # & ((state[:, Col.TYPE] == Types.CAST).sum() < 10)
    not_full = new_inventories.sum(axis=1) <= 10
    return np.arange(0, len(state), 1)[
        craftable & ((~is_learn & not_full) | (is_learn & learnable))
    ]


def predict(user_action, state, inventory):
    """

    Args:
        user_action: the action executed by our player
        other_action: the action executed by the other player
        state: the state of the game a step t
        new_inventory: the inventory of both players at step t+1

    Returns: the state of the game at t+1 an the inventory of both players at t+1
    """
    new_inventory = inventory.copy()
    new_state = state.copy()
    action_type = new_state[user_action, Col.TYPE]
    if (action_type == Types.CAST) or (
            action_type == Types.BREW
    ):
        # just update the inventory and set it as not castable
        new_inventory += new_state[user_action, Col.D1 : Col.PRICE + 1]
        # set castable to false
        new_state[user_action, Col.CASTABLE] = False
    elif action_type == Types.LEARN:
        new_inventory[0] += (
            new_state[user_action, Col.TAX] - new_state[user_action, Col.TOME_INDEX]
        )
        new_state[:user_action, Col.TAX] += new_state[:user_action, Col.TYPE] == Types.LEARN
        # change type
        new_state[user_action, Col.TYPE] = Types.CAST
        # set castable
        new_inventory += new_state[user_action, Col.D1 : Col.PRICE + 1]
        new_inventory[0] += new_state[user_action, Col.TAX]
        new_state[user_action, Col.TAX:-1] = 0
        # put it back
    elif action_type == Types.REST:
        # new_state[:, Col.CASTABLE] = np.where(new_state[:, Col.TYPE] == Types.CAST, 1, new_state[:, Col.TYPE])
        new_state[:, Col.CASTABLE] = True
    # new_inventory[:4] = np.clip(new_inventory[:4], 0, 10)
    return new_state, new_inventory


def minmax(state, inventory, depth):
    """
    Args:
        state: state of the game, both players
        inventory: inventory, both players
        depth: the depth of the recusion used in the algorithm

    Returns: the best (long term) score and the immediate action that lead to this score.

    """
    # 1. termination criterion
    if depth >= max_depth:
        return score_state(state, inventory), None
    actions = available_actions(state, inventory)
    if (state[actions, Col.TYPE] == Types.BREW).any():
        actions_values = state[actions]
        brews = actions_values[actions_values[:, Col.TYPE] == Types.BREW]
        # score = brews[:, Col.PRICE].sum() / (depth+1) + inventory[0, -1]
        action = brews[np.argmax(brews[:, Col.PRICE])]
        score = score_state(state, inventory) + 3 * action[Col.PRICE]
        return score, action
    # 2. exploration
    best_score = -np.inf
    best_action = None
    for action_0 in actions:
        new_state, new_inventory = predict(action_0, state, inventory)
        score, _ = minmax(new_state, new_inventory, depth + 1)
        if score > best_score:
            best_score = score
            best_action = state[action_0]
    return best_score, best_action


def parse_inputs():
    global state, inventory
    action_count = int(input())  # the number of spells and recipes in play
    state = np.vstack(
        [
            [parser(inp) for parser, inp in zip(action_parsers, input().split())]
            for _ in range(action_count)
        ]
        + [[-1, Types.REST, 0, 0, 0, 0, 0, 0, 0, True, True]]
    )
    state = state[state[:, Col.TYPE] != Types.OPPONENT_CAST]
    state[:, Col.CASTABLE] += np.logical_or(
        state[:, Col.TYPE] == Types.BREW, state[:, Col.TYPE] == Types.LEARN
    )
    state[0, Col.PRICE] += 3
    state[1, Col.PRICE] += 1
    inventory = np.array([[int(j) for j in input().split()] for i in range(2)])
    inventory = inventory[0]
    return state, inventory


def trace_execution():
    import time
    n = 50
    r = 0
    for i in range(n):
        t1 = time.thread_time()
        score, action = minmax(state, inventory, 0)
        t2 = time.thread_time()
        r += t2 - t1
    print(f"execution took : {r / n} s")
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.add_function(predict)
    lp.add_function(available_actions)
    lp.add_function(score_state)
    wrapper = lp(minmax)
    wrapper(state, inventory, 0)
    lp.print_stats()
    lp.dump_stats("main.py.lprof")
    exit(0)


if __name__ == '__main__':
    # game loop
    dump_input_at = -1  # if other than -1, stop game at step i and print the input values
    profile = True
    step = 0
    while step != dump_input_at:
        step += 1
        state, inventory = parse_inputs()
        actions = available_actions(state, inventory)
        max_depth = 2 if len(actions) > 5 else 3
        score, action = minmax(state, inventory, 0)
        # if action is None:
        #     print("trigger None", file=sys.stderr)
        #     max_depth = 1
        #     score, action = minmax(state, inventory, 0)
        # print(score, file=sys.stderr)
        print(f"{inv_types[action[1]]} {action[0]}")
        if profile:
            trace_execution()

    while True:
        print(input(), file=sys.stderr)
