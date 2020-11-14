import sys
import numpy as np
from enum import IntEnum


# enum for types
class Types(IntEnum):
    BREW = 0
    CAST = 1
    OPPONENT_CAST = 2
    LEARN = 3
    LEARNED_CAST = 4
    OPPONENT_LEARNED_CAST = 5


# dict used to parse input
types = {
    "BREW": 0,
    "CAST": 1,
    "OPPONENT_CAST": 2,
    "LEARN": 3,
}

inv_types = {
    0: "BREW",
    1: "CAST",
    2: "OPPONENT_CAST",
    3: "LEARN",
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


def score_state(state, inventory):
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
    for _, atype, d0, d1, d2, d3, _, tome_index, _, castable, _ in state:
        new_inventory = np.array([d0, d1, d2, d3]) + inventory[player][:4]
        craftable = new_inventory.min() > 0
        full = new_inventory.sum() > 10
        if craftable and atype == 0: # BREW
            action_filter.append(True)
        elif craftable and castable and player == 0 and atype == 1 and full: # CAST
            action_filter.append(True)
        elif craftable and castable and player == 1 and atype == 2 and full: # OPPONENT_CAST
            action_filter.append(True)
        elif atype == 3 and inventory[player][0] > tome_index: # LEARN
            action_filter.append(True)
        else:
            action_filter.append(False)

    return state[action_filter]


def predict(user_action, other_action, state, inventory):
    """

    Args:
        user_action: the action executed by our player
        other_action: the action executed by the other player
        state: the state of the game a step t
        inventory: the inventory of both players at step t+1

    Returns: the state of the game at t+1 an the inventory of both players at t+1
    """
    state_filter = np.logical_not(np.isin(state[:,0], np.array([user_action[0], other_action[0]])))
    new_state = state[state_filter]
    _, _, d0, d1, d2, d3, price, tome_index, tax, _, _ = user_action
    inventory[0] += [d0 - tome_index + tax, d1, d2, d3, price]
    _, _, d0, d1, d2, d3, price, tome_index, tax, _, _ = other_action
    inventory[1] += [d0 - tome_index + tax, d1, d2, d3, price]
    return new_state, inventory


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
        return score_state(state, inventory), None
    # todo: other ways to end a game
    # 2. exploration
    best_score = -np.inf
    score = best_score
    best_action = None
    for action_0 in available_actions(state, inventory, 0):
        worst_score = np.inf
        for action_1 in available_actions(state, inventory, 1):
            new_state, new_inventory = predict(action_0, action_1, state, inventory)
            score, action = minmax(new_state, new_inventory, depth + 1)
            if worst_score <= worst_score:
                worst_score = score
        if score >= best_score:
            best_score = worst_score
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
    state = np.array(
        [
            [parser(inp) for parser, inp in zip(action_parsers, input().split())]
            for _ in range(action_count)
        ]
    )
    inventory = np.array([[int(j) for j in input().split()] for i in range(2)])
    score, action = minmax(state, inventory, 0)
    #print(score, action, file=sys.stderr)
    if action is None:
        print('REST')
    else:
        print(f'{list(types.keys())[action[1]]} {action[0]}')

while True:
    print(input(), file=sys.stderr)
