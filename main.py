import sys
import numpy as np
from enum import IntEnum
import time


def flushable_memoize_2(memo):
    def memoize_2(f):
        def helper(state, inventory):
            key = hash(state.tobytes()), hash(inventory.tobytes())
            if key not in memo:
                res = f(state, inventory)
                memo[key] = res
                return res
            return memo[key]

        return helper
    return memoize_2


def flushable_memoize_3(memo):
    def memoize_3(f):
        def helper(user_action, state, inventory):
            key = hash(user_action.tobytes()), hash(state.tobytes()), hash(inventory.tobytes())
            if key not in memo:
                res = f(user_action, state, inventory)
                memo[key] = res
                return res
            return memo[key]

        return helper
    return memoize_3


def flushable_memoize_4(memo):
    def memoize_3(f):
        def helper(state, inventory, depth, max_depth, t0):
            key = hash(state.tobytes()), hash(inventory.tobytes()), depth, max_depth
            if key not in memo:
                res = f(state, inventory, depth, max_depth, t0)
                memo[key] = res
                return res
            return memo[key]

        return helper
    return memoize_3

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
    lambda x: max(0, int(x)),
    lambda x: max(0, int(x)),
    lambda x: x != "0",
    lambda x: x != "0",
]

score_state_memo = dict()
weights = np.array([0.5, 1.1, 1.1, 1.1])


@flushable_memoize_2(score_state_memo)
def score_state(state, inventory):
    """
    Args:
        state: state of the game
        inventory: inventory of the 2 players

    Returns: the score describing the "goodness" of a state

    """
    castable_brews = (state[:, Col.TYPE] == Types.BREW) & state[:, Col.CASTABLE]
    return (((
        np.clip(
            state[castable_brews, 2:6] + inventory[:-1],
            None,
            0,
        ) * weights)
        .sum(axis=1) * state[castable_brews, Col.PRICE])
        .mean()
        + 10 * ((inventory[-1] - opponent_score) + last_brew * inventory[1:-1].sum())
    )


available_actions_memo = dict()


@flushable_memoize_2(available_actions_memo)
def available_actions(state, inventory):
    """
    Args:
        state: the state of the game (after removing other player actions) the inventory of the player
        inventory: the inventory of the player

    Returns: the list of available action, as a subset of state

    """
    new_inventories = state[:, 2:6] + inventory[:4]
    craftable = (new_inventories.min(axis=1) >= 0) & (state[:, Col.CASTABLE] == 1)
    is_learn = state[:, Col.TYPE] == Types.LEARN
    learnable = (
        state[:, Col.TOME_INDEX] <= inventory[0]
    ) & (state[:, Col.TAX] + inventory[:4].sum() <= 10)
    not_full = new_inventories.sum(axis=1) <= 10
    return np.arange(0, len(state), 1)[
        ((~is_learn & craftable & not_full) | (is_learn & learnable))
    ]


predict_memo = dict()


@flushable_memoize_3(predict_memo)
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
        new_state[user_action, Col.CASTABLE] = new_state[user_action, Col.REPEATABLE]
        ninv = new_inventory + new_state[user_action, Col.D1: Col.PRICE + 1]
        if new_state[user_action, Col.REPEATABLE] and (ninv >= 0).all() and (ninv[:-1].sum() < 10):
            new_inventory = ninv
            # remove tax and set not castable
            new_state[user_action, -1] = 2
        else:
            new_state[user_action, Col.CASTABLE] = False  # new_state[user_action, Col.REPEATABLE]
    elif action_type == Types.LEARN:
        new_inventory[0] += (
            new_state[user_action, Col.TAX] - new_state[user_action, Col.TOME_INDEX]
        )
        new_state[:user_action, Col.TAX] += new_state[:user_action, Col.TYPE] == Types.LEARN
        # apply learned
        hyp_inv = new_inventory + new_state[user_action, Col.D1: Col.PRICE+1]
        if (hyp_inv >= 0).all():
            new_inventory = hyp_inv
            # earn tax
            new_inventory[0] += new_state[user_action, Col.TAX]
            # change type
            new_state[user_action, Col.TYPE] = Types.CAST
            # remove tax and set not castable
            new_state[user_action, Col.TAX] = 0
            new_state[user_action, Col.CASTABLE] = 1
            hyp_inv = new_inventory + new_state[user_action, Col.D1: Col.PRICE + 1]
            if new_state[user_action, Col.REPEATABLE] and (hyp_inv >= 0).all():
                new_inventory = hyp_inv
                # remove tax and set not castable
                new_state[user_action, -1] = 2
        else:
            # earn tax
            new_inventory[0] += new_state[user_action, Col.TAX]
            # change type
            new_state[user_action, Col.TYPE] = Types.CAST
            # set castable
            new_state[user_action, Col.TAX] = 0
            new_state[user_action, Col.CASTABLE] = 1
    elif action_type == Types.REST:
        # new_state[:, Col.CASTABLE] = np.where(new_state[:, Col.TYPE] == Types.CAST, 1, new_state[:, Col.TYPE])
        new_state[:, Col.CASTABLE] = True
    # new_inventory[:4] = np.clip(new_inventory[:4], 0, 10)
    return new_state, new_inventory


minmax_memo = dict()


@flushable_memoize_4(minmax_memo)
def minmax(state, inventory, depth, max_depth, t0):
    """
    Args:
        state: state of the game, both players
        inventory: inventory, both players
        depth: the depth of the recusion used in the algorithm

    Returns: the best (long term) score and the immediate action that lead to this score.

    """
    # 1. termination criterion
    if time.time() - t0 > 0.049: return -np.inf, None
    current_score = score_state(state, inventory)
    if depth >= max_depth:
        return current_score, None
    actions = available_actions(state, inventory)
    # 2. exploration
    best_score = -np.inf
    best_action = None
    for action_0 in actions:
        if time.time() - t0 > 0.0485: return -np.inf, None
        new_state, new_inventory = predict(action_0, state, inventory)
        score, next_best_action = minmax(new_state, new_inventory, depth + 1, max_depth, t0)
        if score is not None and score > best_score:
            if time.time() - t0 > 0.0485: return -np.inf, None
            best_score = score
            best_action = state[action_0].copy()
            best_action[-1] = new_state[action_0, -1]
            if next_best_action is not None and best_action[0] and best_action[0] == next_best_action[0]:
                best_action[-1] += next_best_action[-1]
    return 0.95 * best_score + 0.05 * current_score, best_action


def trace_execution():
    n = 100
    r = 0
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp.add_function(predict)
    lp.add_function(available_actions)
    lp.add_function(score_state)
    wrapper = lp(minmax)
    for i in range(n):
        t1 = time.thread_time()
        wrapper(state, inventory, 0, max_depth, time.time())
        t2 = time.thread_time()
        r += t2 - t1
    print(f"execution took : {r / n} s")
    lp.print_stats()
    lp.dump_stats("main.py.lprof")
    exit(0)


if __name__ == '__main__':
    # game loop
    dump_input_at = -1  # if other than -1, stop game at step i and print the input values
    profile = False
    step = 0
    last_brew = False
    nb_brews = 0
    opponent_brews = 0
    opponent_score = 0
    opponent_inv_sum = 0
    while True:  # step != dump_input_at:
        step += 1
        action_count = int(input())  # the number of spells and recipes in play
        t0 = time.time()
        state = np.array(
            list(
                filter(  # drop opponent actions
                    lambda x: x[1] != Types.OPPONENT_CAST,
                    (
                        [parser(inp) for parser, inp in zip(action_parsers, input().split())]
                        for _ in range(action_count)
                    )
                ),
                # key=lambda x: x[Col.TYPE]
            ) + [[-1, Types.REST, 0, 0, 0, 0, 0, 0, 0, True, False]],  # adds a row for the rest action

        )
        state[:, Col.CASTABLE] += np.logical_or(  # set learn and brew as castable to simplify available_action
            state[:, Col.TYPE] == Types.BREW, state[:, Col.TYPE] == Types.LEARN
        )
        state[0, Col.PRICE] += 3 * (state[0, Col.TAX] > 0)  # account first order bonus
        state[1, Col.PRICE] += 1 * (state[1, Col.TAX] > 0)  # account second order bonus
        inventory = np.array([[int(j) for j in input().split()] for i in range(2)])

        if opponent_score != inventory[1, -1]:
            opponent_brews += 1
        opponent_score = inventory[1, -1]
        # if last_brew:
        #     opponent_inv_sum = inventory[1, 1:-1].sum()
        inventory = inventory[0]  # drop opponent inventory
        max_depth = 1
        best_action = None
        best_score = -np.inf
        while (time.time() - t0) < 0.035:
            score, action = minmax(state, inventory, 0, max_depth, t0)
            if score >= best_score and action is not None:
                best_action = action
            print(f"depth: {max_depth}, time: {time.time() - t0}, score: {score}, action:{best_action[0]}", file=sys.stderr)
            max_depth += 1
        # print(f"execution time: {(time.time() - t0)*1000}ms", file=sys.stderr)
        print(f"{inv_types[best_action[1]]} {best_action[0]} {max(1, best_action[-1])}")
        if best_action[1] == 0:
            nb_brews += 1
            last_brew = (nb_brews == 5) or (opponent_brews == 5)
        if (time.time() - t0) < 0.040:
            predict_memo.clear()
            available_actions_memo.clear()
            score_state_memo.clear()
            minmax_memo.clear()

        # if profile:
        #     trace_execution()

    while True:
        print(input(), file=sys.stderr)
