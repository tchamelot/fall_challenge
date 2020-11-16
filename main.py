import sys
import numpy as np
from enum import IntEnum
import time


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
types = {
    "BREW": 0,
    "CAST": 1,
    "OPPONENT_CAST": 2,
    "LEARN": 3,
    "REST":4
}

inv_types = {
    0: "BREW",
    1: "CAST",
    2: "OPPONENT_CAST",
    3: "LEARN",
    4: "REST"
}

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


def score_state(state, inventory, depth):
    """
    Args:
        state: state of the game
        inventory: inventory of the 2 players

    Returns: the score describing the "goodness" of a state

    """
    # score = np.clip(
    #     state[state[:, Col.TYPE] == Types.BREW, Col.D1:Col.D4+1] + inventory[0, :-1]
    #     , None, 0).sum(axis=1).min() + inventory[0, -1]
    # print((" " * depth) + "|"+str(score))
    return np.clip(
        state[state[:, Col.TYPE] == Types.BREW, Col.D1:Col.D4+1] + inventory[0, :-1]
        , None, 0).sum(axis=1).min() + inventory[0, -1]


def available_actions(state, inventory, player):
    """
    Args:
        state: the state of the game (after removing other player actions) the inventory of the player
        inventory: the inventory of the player

    Returns: the list of available action, as a subset of state

    """
    action_filter = list()
    new_inventories = state[:, Col.D1:Col.D4+1] + inventory[0, :4]
    craftable = (new_inventories.min(axis=1) >= 0) & (state[:, Col.CASTABLE] == 1)
    is_learn = state[:, Col.TYPE] == Types.LEARN
    learnable = state[:, Col.TOME_INDEX] <= inventory[0, 0]
    not_full = new_inventories.sum(axis=1) <= 10
    res = np.arange(0, len(state), 1)[
        craftable & ((~is_learn & not_full) | (is_learn & learnable))
    ]
    # for _, atype, d0, d1, d2, d3, _, tome_index, tax, castable, _ in state:
    #     new_inventory = np.array([d0, d1, d2, d3]) + inventory[player][:4]
    #     craftable = new_inventory.min() >= 0
    #     not_full = new_inventory.sum() <= 10
    #     if atype == Types.REST:
    #         action_filter.append(True)
    #     elif craftable and atype == Types.BREW:  # BREW
    #         action_filter.append(True)
    #     elif craftable and castable and player == 0 and atype == Types.CAST and not_full: # CAST
    #         action_filter.append(True)
    #     elif craftable and castable and player == 1 and atype == Types.OPPONENT_CAST and not_full: # OPPONENT_CAST
    #         action_filter.append(True)
    #     elif craftable and atype == Types.LEARN and inventory[player][0] >= tome_index: # LEARN
    #         action_filter.append(True)
    #     else:
    #         action_filter.append(False)
    # res2 = np.arange(0, len(state), 1)[action_filter]
    # assert res == res2
    return res


def predict(user_action, other_action, state, inventory):
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
    user_action = new_state[user_action]
    # other_action = new_state[other_action]
    # user_action = user_action.copy()
    # # other_action = other_action.copy()
    # state_filter = np.logical_not(state[:, 0] == user_action[0])
    # # state_filter = np.logical_not(np.isin(state[:,0], np.array([user_action[0], other_action[0]])))
    # # remove played actions
    # new_state = new_state[state_filter]
    # brew and cast works the same way
    if (user_action[Col.TYPE] == Types.CAST) or (user_action[Col.TYPE] == Types.BREW):
        # just update the inventory and set it as not castable
        new_inventory[0] += user_action[Col.D1:Col.PRICE+1]
        # set castable to false
        user_action[Col.CASTABLE] = False
        # new_state = np.vstack([new_state, user_action])
    if user_action[Col.TYPE] == Types.LEARN:
        new_inventory[0, 0] += user_action[Col.TAX] - user_action[Col.TOME_INDEX]
        # change type
        user_action[Col.TYPE] = Types.CAST
        # set castable
        new_inventory[0] += user_action[Col.D1:Col.PRICE + 1]
        user_action[Col.CASTABLE] = False
        # put it back
        # new_state = np.vstack([new_state, user_action])
    if user_action[Col.TYPE] == Types.REST:
        new_state[:, Col.CASTABLE] = np.where(new_state[:, Col.TYPE] == Types.CAST, 1, new_state[:, Col.TYPE])
        # new_state = np.vstack([new_state, user_action])
    new_inventory[0, :4] = np.clip(new_inventory[0, :4], 0, 10)

    # # # do the same for opponent
    # if (other_action[Col.TYPE] == Types.CAST) or (other_action[Col.TYPE] == Types.BREW):
    #     # just update the inventory and set it as not castable
    #     new_inventory[1] += other_action[Col.D1:Col.PRICE+1]
    #     # set castable to false
    #     other_action[Col.CASTABLE] = False
    #     new_state = np.vstack([new_state, other_action])
    # if other_action[Col.TYPE] == Types.LEARN:
    #     new_inventory[1, 0] += other_action[Col.TAX] - other_action[Col.TOME_INDEX]
    #     # change type
    #     other_action[Col.TYPE] = Types.CAST
    #     # set castable
    #     other_action[Col.CASTABLE] = True
    #     # put it back
    #     new_state = np.vstack([new_state, other_action])
    # if other_action[Col.TYPE] == Types.REST:
    #     new_state[:, Col.CASTABLE] = np.where(new_state[:, Col.TYPE] == Types.CAST, 1, new_state[:, Col.TYPE])
    #     new_state = np.vstack([new_state, other_action])
    # new_inventory[1, :4] = np.clip(new_inventory[1, :4], 0, 10)
    return new_state, new_inventory


max_depth = 3


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
        return score_state(state, inventory, depth), None
    actions = available_actions(state, inventory, 0)
    if np.isin(Types.BREW, state[actions, Col.TYPE]):
        actions_values = state[actions]
        brews = actions_values[actions_values[:, Col.TYPE] == Types.BREW]
        # score = brews[:, Col.PRICE].sum() / (depth+1) + inventory[0, -1]
        action = brews[np.argmax(brews[:, Col.PRICE])]
        score = score_state(state, inventory, depth) + action[Col.PRICE]
        return score, action
    # todo: other ways to end a game
    # 2. exploration
    best_score = -np.inf
    best_action = None
    for action_0 in actions:
        # worst_score = np.inf
        # for action_1 in available_actions(state, inventory, 1):
        # new_state, new_inventory = predict(action_0, None, state, inventory)
        # print((" "*depth) + "|"+"{} {}".format(state[action_0][0], state[action_0][1]))
        score, _ = minmax(*predict(action_0, None, state, inventory), depth + 1)
        # if score < worst_score:
        #     worst_score = score
        if score > best_score:
            best_score = score
            best_action = state[action_0]
    return best_score, best_action


# game loop
dump_input_at = -1  # if other than -1, stop game at step i and print the input values
step = 0
while True:
    if step == dump_input_at:
        break
    step += 1
    action_count = int(input())  # the number of spells and recipes in play
    state = np.vstack(
        [
            [parser(inp) for parser, inp in zip(action_parsers, input().split())]
            for _ in range(action_count)
        ] +
        [[-1, Types.REST, 0,0,0,0, 0,0,0,True,True]]
    )
    state = state[state[:, Col.TYPE] != Types.OPPONENT_CAST]
    state[:, Col.CASTABLE] += np.logical_or(state[:, Col.TYPE] == Types.BREW, state[:, Col.TYPE] == Types.LEARN)
    state[0, Col.PRICE] += 3
    state[1, Col.PRICE] += 1
    inventory = np.array([[int(j) for j in input().split()] for i in range(2)])
    inventory = inventory[:1]
    actions = available_actions(state, inventory, 0)
    max_depth = 2 if len(actions) > 5 else 3
    # r = 0
    # for i in range(10):
    #     t1 = time.thread_time()
    #     score, action = minmax(state, inventory, 0)
    #     t2 = time.thread_time()
    #     r += t2 - t1
    # print(f'execution took : {r/10} s', file=sys.stderr)
    score, action = minmax(state, inventory, 0)
    print(f'{inv_types[action[1]]} {action[0]}')

while True:
    print(input(), file=sys.stderr)
