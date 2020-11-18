import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import pool

# tome\.add\(new\nTomeSpell\(new\nRecipe\(((.+(,\s*)?)+)\)\)\);
from main import Types, score_state, minmax

base_spells = [
    [2, 0, 0, 0],
    [-1, 1, 0, 0],
    [0, -1, 1, 0],
    [0, 0, -1, 1]
]

spells = [
    [-3, 0, 0, 1],
    [3, -1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [3, 0, 0, 0],
    [2, 3, -2, 0],
    [2, 1, -2, 1],
    [3, 0, 1, -1],
    [3, -2, 1, 0],
    [2, -3, 2, 0],
    [2, 2, 0, -1],
    [-4, 0, 2, 0],
    [2, 1, 0, 0],
    [4, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 2, 0, 0],
    [1, 0, 1, 0],
    [-2, 0, 1, 0],
    [-1, -1, 0, 1],
    [0, 2, -1, 0],
    [2, -2, 0, 1],
    [-3, 1, 1, 0],
    [0, 2, -2, 1],
    [1, -3, 1, 1],
    [0, 3, 0, -1],
    [0, -3, 0, 2],
    [1, 1, 1, -1],
    [1, 2, -1, 0],
    [4, 1, -1, 0],
    [-5, 0, 0, 2],
    [-4, 0, 1, 1],
    [0, 3, 2, -2],
    [1, 1, 3, -2],
    [-5, 0, 3, 0],
    [-2, 0, -1, 2],
    [0, 0, -3, 3],
    [0, -3, 3, 0],
    [-3, 3, 0, 0],
    [-2, 2, 0, 0],
    [0, 0, -2, 2],
    [0, -2, 2, 0],
    [0, 0, 2, -1],
]

# deliveries\.add\(new\nDeliverySpell\(new\nRecipe\(((.+(,\s*)?)+)\),\s+(\w+)\)\);
deliveries = [
    [2, 2, 0, 0, 6],
    [3, 2, 0, 0, 7],
    [0, 4, 0, 0, 8],
    [2, 0, 2, 0, 8],
    [2, 3, 0, 0, 8],
    [3, 0, 2, 0, 9],
    [0, 2, 2, 0, 10],
    [0, 5, 0, 0, 10],
    [2, 0, 0, 2, 10],
    [2, 0, 3, 0, 11],
    [3, 0, 0, 2, 11],
    [0, 0, 4, 0, 12],
    [0, 2, 0, 2, 12],
    [0, 3, 2, 0, 12],
    [0, 2, 3, 0, 13],
    [0, 0, 2, 2, 14],
    [0, 3, 0, 2, 14],
    [2, 0, 0, 3, 14],
    [0, 0, 5, 0, 15],
    [0, 0, 0, 4, 16],
    [0, 2, 0, 3, 16],
    [0, 0, 3, 2, 17],
    [0, 0, 2, 3, 18],
    [0, 0, 0, 5, 20],
    [2, 1, 0, 1, 9],
    [0, 2, 1, 1, 12],
    [1, 0, 2, 1, 12],
    [2, 2, 2, 0, 13],
    [2, 2, 0, 2, 15],
    [2, 0, 2, 2, 17],
    [0, 2, 2, 2, 19],
    [1, 1, 1, 1, 12],
    [3, 1, 1, 1, 14],
    [1, 3, 1, 1, 16],
    [1, 1, 3, 1, 18],
    [1, 1, 1, 3, 20]
]

brews = pd.DataFrame(
    np.hstack(
        [
            np.expand_dims(np.arange(0, len(deliveries), 1), axis=1),
            np.zeros((len(deliveries), 1)),
            np.array(deliveries),
            np.zeros((len(deliveries), 4))
        ]
    ),
    columns=[
        "id", "type", "d1", "d2", "d3", "d4", "price", "tome", "tax", "castable", "repeatable"
    ]
)
brews['castable'] = 1

casts = pd.DataFrame(
    np.hstack(
        [
            np.expand_dims(np.arange(0, len(spells), 1), axis=1),
            np.zeros((len(spells), 1)),
            np.array(spells),
            np.zeros((len(spells), 5))
        ]
    ),
    columns=[
        "id", "type", "d1", "d2", "d3", "d4", "price", "tome", "tax", "castable", "repeatable"
    ]
)
casts['repeatable'] = (casts[["d1", "d2", "d3", "d4"]] < 0).any(axis=1)
# castable will be random

base_casts = pd.DataFrame(
    np.hstack(
        [
            np.expand_dims(np.arange(0, len(base_spells), 1), axis=1),
            np.ones((len(base_spells), 1)),
            np.array(base_spells),
            np.zeros((len(base_spells), 5))
        ]
    ),
    columns=[
        "id", "type", "d1", "d2", "d3", "d4", "price", "tome", "tax", "castable", "repeatable"
    ]
)
#castable will be random


def generate_state(pool, batch_size=255):
    while True:
        x_s = []
        x_lengths = [0]
        y_s = []
        def single_batch():
            # draw brews
            current_brews = brews.sample(5).values.astype(int)
            # take base_spells
            current_player_casts = base_casts.values.astype(int)
            # sample casts
            deck_casts = casts.sample(np.random.randint(6, 20)).values.astype(int)
            # select learned ones
            player_cast_mask = np.random.rand(len(deck_casts)) < 0.5
            player_cast = deck_casts[player_cast_mask]
            # randomly set casts as castable
            current_player_casts = np.vstack(
                [
                    current_player_casts,
                    player_cast
                ]
            )
            current_player_casts[:, -2] = np.random.rand(len(current_player_casts)) < 0.5
            current_player_casts[:, 1] = Types.CAST
            # create learnables (plausible tomeindex & tax !)
            player_learned_spells = deck_casts[~player_cast_mask]
            t1, t2 = np.random.randint(0, 2, 2)
            if len(player_learned_spells) > 0:
                player_learned_spells[0, -3] = t1 + t2
            if len(player_learned_spells) > 1:
                player_learned_spells[1, -3] = t2
            # compute tomeindex
            player_learned_spells[:, -4] = np.arange(0, len(player_learned_spells))
            # set castable
            player_learned_spells[:, -2] = True
            # set type
            player_learned_spells[:, 1] = Types.LEARN
            # compute inventory
            items = np.random.randint(0, 5+3, 10)
            inventory = np.array([(items == i).sum() for i in range(4)] + [np.random.randint(0, 50)])
            state = np.vstack(
                [
                    current_brews,
                    player_learned_spells,
                    current_player_casts,
                    [[-1, Types.REST, 0, 0, 0, 0, 0, 0, 0, True, True]]
                ]
            )
            x_s.append(
                np.hstack(
                    [
                        state,
                        np.repeat(np.expand_dims(inventory, axis=0), len(state), axis=0)
                    ]
                )
            )
            x_lengths.append(x_lengths[-1] + len(state))
            y_s.append(minmax(state, inventory, depth=1, t0=None, max_depth=3)[0])

        yield tf.RaggedTensor.from_row_splits(np.vstack(x_s), x_lengths), tf.stack(y_s)


if __name__ == '__main__':
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(None, 16)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2)),
            tf.keras.layers.Dense(1)
        ]
    )
    # inputs = tf.keras.Input(shape=(None, 16))
    # my_layer = tf.keras.layers.Dense(64)
    # td_outputs = tf.keras.layers.TimeDistributed(my_layer)(inputs)
    # outputs = tf.reduce_sum(td_outputs, axis=1)
    # model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.mean_absolute_error)
    print(model.input_shape)
    print(model.output_shape)
    gen = generate_state(2000)
    # dataset = np.array(list(zip(*[gen.send(None) for i in range(100)])))
    for i in range(500):
        x, y = gen.send(None)
        print(model.train_on_batch(x, y))
    # model.fit_generator(gen, steps_per_epoch=10, epochs=1)
    states = np.array((gen.send(None) for i in range(3)))
    print(states)
    # print(inventory)
