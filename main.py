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
    # Calcul de coup en distance
    # coup = price - d0 * 1 - d1 *2 - d2 *3 -d3 *4
    cost = np.array([1, 2, 3, 4, 1])
    scores = np.array([])
    for _, atype, d0, d1, d2, d3, price, _, _, _, _ in state:
        recipe = np.array([d0, d1, d2, d3, price])
        if atype == 0:
            user_score = ((inventory[0] + recipe) * cost).sum()
            other_score = ((inventory[1] + recipe) * cost).sum()
            scores = np.append(scores, user_score - other_score)
        else:
            scores = np.append(scores, 0)

    return np.mean(scores)


def available_actions(state, inventory, player):
    """
    Args:
        state: the state of the game (after removing other player actions) the inventory of the player
        inventory: the inventory of the player

    Returns: the list of available action, as a subset of state

    """
    action_filter = list()
    for _, atype, d0, d1, d2, d3, _, tome_index, tax, castable, _ in state:
        new_inventory = np.array([d0, d1, d2, d3]) + inventory[player][:4]
        craftable = new_inventory.min() >= 0
        not_full = new_inventory.sum() < 10
        if atype == Types.REST:
            action_filter.append(True)
        elif craftable and atype == Types.BREW:  # BREW
            action_filter.append(True)
        elif craftable and castable and player == 0 and atype == Types.CAST and not_full: # CAST
            action_filter.append(True)
        elif craftable and castable and player == 1 and atype == Types.OPPONENT_CAST and not_full: # OPPONENT_CAST
            action_filter.append(True)
        elif atype == Types.LEARN and inventory[player][0] >= tome_index: # LEARN
            action_filter.append(True)
        else:
            action_filter.append(False)

    return state[action_filter]


def predict(user_action, other_action, state, new_inventory):
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
    user_action = user_action.copy()
    # other_action = other_action.copy()
    state_filter = np.logical_not(state[:, 0] == user_action[0])
    # state_filter = np.logical_not(np.isin(state[:,0], np.array([user_action[0], other_action[0]])))
    # remove played actions
    new_state = new_state[state_filter]
    # brew and cast works the same way
    if (user_action[Col.TYPE] == Types.CAST) or (user_action[Col.TYPE] == Types.BREW):
        # just update the inventory and set it as not castable
        new_inventory[0] += user_action[Col.D1:Col.PRICE+1]
        # set castable to false
        user_action[Col.CASTABLE] = False
        new_state = np.vstack([new_state, user_action])
    if user_action[Col.TYPE] == Types.LEARN:
        new_inventory[0, 0] += user_action[Col.TAX] - user_action[Col.TOME_INDEX]
        # change type
        user_action[Col.TYPE] = Types.CAST
        # set castable
        user_action[Col.CASTABLE] = True
        # put it back
        new_state = np.vstack([new_state, user_action])
    if user_action[Col.TYPE] == Types.REST:
        new_state[:, Col.CASTABLE] = np.where(new_state[:, Col.TYPE] == Types.CAST, 1, new_state[:, Col.TYPE])
        new_state = np.vstack([new_state, user_action])

    # # do the same for opponent
    # if (other_action[Col.TYPE] == Types.CAST) or (other_action[Col.TYPE] == Types.BREW):
    #     # just update the inventory and set it as not castable
    #     new_inventory[1] += other_action[Col.D1:Col.PRICE+1]
    #     # set castable to false
    #     other_action[Col.CASTABLE] = False
    #     new_state = np.vstack([new_state, other_action])
    # if other_action[Col.TYPE] == Types.LEARN:
    #     new_inventory[1, 0] += other_action[Col.TAX] - other_action[Col.TOME_INDEX]
    #     # change type
    #     other_action[Col.TYPE] = Types.OPPONENT_CAST
    #     # set castable
    #     other_action[Col.CASTABLE] = True
    #     # put it back
    #     new_state = np.vstack([new_state, other_action])
    # if other_action[Col.TYPE] == Types.REST:
    #     new_state[:, Col.CASTABLE] = np.where(new_state[:, Col.TYPE] == Types.OPPONENT_CAST, 1, new_state[:, Col.TYPE])
    #     new_state = np.vstack([new_state, other_action])
    # # _, _, d0, d1, d2, d3, price, tome_index, tax, _, _ = user_action
    # # inventory[0] += [d0 - tome_index + tax, d1, d2, d3, price]
    # # _, _, d0, d1, d2, d3, price, tome_index, tax, _, _ = other_action
    # # inventory[1] += [d0 - tome_index + tax, d1, d2, d3, price]
    return new_state, new_inventory


max_depth = 2


def minmax(state, inventory, depth):
    """
    Args:
        state: state of the game, both players
        inventory: inventory, both players
        depth: the depth of the recusion used in the algorithm

    Returns: the best (long term) score and the immediate action that lead to this score.

    """
    # 1. termination criterion
    if depth == max_depth:
        return score_state(state, inventory, depth), None
    actions = available_actions(state, inventory, 0)
    if np.isin(Types.BREW, actions[:, Col.TYPE]):
        brews = actions[actions[:, Col.TYPE] == Types.BREW]
        # score = brews[:, Col.PRICE].sum() / (depth+1) + inventory[0, -1]
        score = score_state(state, inventory, depth)
        action = brews[np.argmax(brews[:, Col.PRICE])]
        return score, action
    # todo: other ways to end a game
    # 2. exploration
    best_score = -np.inf
    best_action = None
    for action_0 in actions:
        # worst_score = np.inf
        # for action_1 in available_actions(state, inventory, 1):
        new_state, new_inventory = predict(action_0, None, state, inventory)
        score, action = minmax(new_state, new_inventory, depth + 1)
            # if score < worst_score:
            #     worst_score = score
        if score > best_score:
            best_score = score
            best_action = action_0
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
        [[0, Types.REST, 0,0,0,0, 0,0,0,True,True]]
    )
    inventory = np.array([[int(j) for j in input().split()] for i in range(2)])
    score, action = minmax(state, inventory, 0)
    print(f'{inv_types[action[1]]} {action[0]} {score}')

while True:
    print(input(), file=sys.stderr)
