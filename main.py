import sys
import numpy as np
from enum import IntEnum


class Types(IntEnum):
    BREW = 0
    CAST = 1
    LEARN = 2
    OPPONENT_CAST = 3
    LEARNED_CAST = 4
    OPPONENT_LEARNED_CAST = 5


types = {
    "BREW": 0,
    "CAST": 1,
    "LEARN": 2,
    "OPPONENT_CAST": 3,
}

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
    return inventory[0, -1] - inventory[1, -1]


def available_actions(state, inventory, player):
    return []


def predict(user_action, other_action, state, inventory):
    if user_action[1] != Types.LEARN:
        pass
    new_state = state
    new_inventory = inventory
    return new_state, new_inventory


max_depth = 2


def minmax(state, inventory, depth):
    # 1. termination criterion
    if depth == max_depth:
        return score_state(state, inventory), None
    # todo: other ways to end a game
    # 2. exploration
    best_score = -np.inf
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
dump_input_at = 16  # if other than -1, stop game at step i and print the input values
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
    print("REST")
while True:
    print(input(), file=sys.stderr)
