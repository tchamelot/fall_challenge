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

    def action(self, orders):
        """
        Select the action to do
        """
        target = orders.best(self.inventory)
        print(f'BREW {target}')


class Orders:
    """
    Base class for client's orders
    """
    def __init__(self):
        self.orders = {}

    def parse(self):
        """
        Read orders from stdin
        """
        nb_action = int(input())
        self.orders = {}
        for _ in range(nb_action):
            aid, atype, d0, d1, d2, d3, price, _, _, _, _ = input().split()
            self.orders[aid] = {
                    'type': atype,
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
            if (inventory - order['recipe']).min() >= 0:
                return order_id

        return {}



def main():
    """
    Main loop
    """
    ally = Witch()
    enemy = Witch()
    orders = Orders()

    while True:
        orders.parse()
        ally.parse()
        enemy.parse()
        ally.action(orders)


if __name__ == '__main__':
    sys.exit(main())
