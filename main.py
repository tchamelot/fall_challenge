"""
Python script for the codingame fall challenge 2020
"""
import sys
import numpy as np


class Witch:
    """
    Base class for the players
    """
    def __init__(self):
        self.inventory = []
        self.rubis = 0

    def parse(self):
        """
        Read witch from stdin
        """
        player = input().split()
        self.inventory = np.array([int(x) for x in player[:4]])
        self.rubis = int(player[-1])

    def action(self, actions):
        """
        Select the action to do
        """
        target = actions.best(self.inventory)
        print(target)


class Actions:
    """
    Base class for actions
    """
    def __init__(self):
        self.orders = {}
        self.ally_cast = {}
        self.enemy_cast = {}

    def parse(self):
        """
        Read orders from stdin
        """
        nb_action = int(input())
        self.orders = {}
        self.ally_cast = {}
        self.enemy_cast = {}
        for _ in range(nb_action):
            aid, atype, d0, d1, d2, d3, price, _, _, castable, _ = input().split()
            if castable == "1" and atype == "CAST":
                self.ally_cast[aid] = np.array([int(d0), int(d1), int(d2), int(d3)])
            elif castable == "1" and atype == "OPPONENT_CAST":
                self.enemy_cast[aid] = np.array([int(d0), int(d1), int(d2), int(d3)])
            elif atype == "BREW":
                self.orders[aid] = {
                    'recipe': np.array([int(d0), int(d1), int(d2), int(d3)]),
                    'price': int(price),
                    }

    def best(self, inventory):
        """
        return the best order given an inventory
        """
        self.orders = dict(sorted(
            self.orders.items(),
            key=lambda item: item[1]['price'],
            reverse=True))

        for order_id, order in self.orders.items():
            if (inventory + order['recipe']).min() >= 0:
                return f'BREW {order_id}'

        for cast_id, cast in self.ally_cast.items():
            new_inventory = inventory + cast
            # print(new_inventory, file=sys.stderr)
            if new_inventory.sum() <= 10 and new_inventory.min() >= 0:
                return f'CAST {cast_id}'

        return 'REST'



def main():
    """
    Main loop
    """
    ally = Witch()
    enemy = Witch()
    actions = Actions()

    while True:
        actions.parse()
        ally.parse()
        enemy.parse()
        ally.action(actions)


if __name__ == '__main__':
    sys.exit(main())
