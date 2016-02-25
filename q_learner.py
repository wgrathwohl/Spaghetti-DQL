"""
Implements the bones of the q-learning algorithm
maintains the replay memory 
"""

from game_engine import *
import numpy as np
from trainer import LiveQModel
import networks
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time
import matplotlib.animation as animation

TRAIN_DIR = "/tmp/q_learning_training_medium"

class Q_Function:
    def __init__(self, possible_actions):
        self.possible_actions = possible_actions
    def generate_q_values(self, state_tensor):
        """
        Feeds state tensor through q network and returns
        vector of q values
        """

class DQLearning:
    def __init__(self,
        memory_size=1e6, num_frames=4, num_channels=3,
        num_food=1, num_enemies=5, game_size=25,
        player_size=2, object_size=1, 
        possible_actions=MOVE_INDS,
        epsilon=.1, decay_factor=.9,
        decay_steps=1e4, initial_learning_rate=.01,
        batch_size=32,
        display=True,
        checkpoint_path=None
    ):
        self.display = display
        self.memory_size = memory_size
        self.game_size = game_size
        self.game_engine = SquareGameEngine(
            num_food, num_enemies, game_size,
            player_size, object_size
        )
        self.batch_size = batch_size
        self.replay = []

        self.possible_actions = possible_actions
        self.reverse_actions = {action: i for i, action in enumerate(possible_actions)}
    
        self.decay_factor = decay_factor
        self.game_size = game_size
        self.player_size = player_size
        self.object_size = object_size
        self.num_food = num_food
        self.num_enemies = num_enemies
        self.QFunc = LiveQModel(
            networks.q_learning_model1, 
            (game_size, game_size), num_channels * num_frames, 
            batch_size,
            possible_actions, TRAIN_DIR,
            tf.train.AdamOptimizer,
            decay_steps, initial_learning_rate, learning_rate_decay_factor=decay_factor,
            checkpoint_path=checkpoint_path
        )
        # used for random exploration
        self.epsilon = epsilon
        # this is used to generate "features"
        # the input to the dqn is the last 4 frames stacked up
        self.frame_buffer = [np.zeros((game_size, game_size, num_channels)) for i in range(num_frames)]

        self.steps = 0

    def step(self, return_im=False):
        current_state_tensor = np.dstack(self.frame_buffer)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.possible_actions)
            #print "Random action: {}".format(action)
        else:
            q_values = self.QFunc.get_q_values(current_state_tensor)
            action_ind = np.argmax(q_values)
            action = self.possible_actions[action_ind]
            if self.display or self.steps % 100 == 0:
                print "Q_values: {}".format(q_values)
                print "Action: {}".format(action)

        # update game state with action
        reward, terminal = self.game_engine.update(action)
        #print "Observed reward: {}, terminal: {}".format(reward, terminal)

        # observe new state
        next_image = self.game_engine.render()
        if self.display:
            im = next_image.astype(np.uint8)
            plt.imshow(im)
            plt.show()
        # update frame buffer
        self.frame_buffer.append(next_image)
        self.frame_buffer = self.frame_buffer[1:]
        next_state_tensor = np.dstack(self.frame_buffer)

       

        # generate transition
        transition = (current_state_tensor, action, reward, terminal, next_state_tensor)
        # add to replay
        if len(self.replay) < self.memory_size:
            self.replay.append(transition)
        else:
            ind = np.random.randint(0, self.memory_size)
            self.replay[ind] = transition

        self.steps += 1
        if terminal:
            DQL.reset_game_engine()

        if return_im:
            return next_image, reward, terminal

    def generate_batch(self):
        sample = random.sample(self.replay, self.batch_size)

        states = np.array([s[0] for s in sample])
        next_states = np.array([s[-1] for s in sample])
        q_values = self.QFunc.get_q_values_multi(next_states)
        # each element of ys is a vector of all zeros except with y
        # in the index of the chosen action
        ys   = np.array([np.array([0. for i in range(len(self.possible_actions))]) for s in sample])
        inds = np.array([np.array([0. for i in range(len(self.possible_actions))]) for s in sample])

        for i, ((s, action, reward, terminal, s_n), qs) in enumerate(zip(sample, q_values)):
            ind = self.reverse_actions[action]
            
            if terminal:
                y = reward
            else:
                y = reward + self.decay_factor * max(qs)

            ys[i][ind]   = y
            inds[i][ind] = 1
        return states, ys, inds

    def apply_batch(self):
        states, ys, inds = self.generate_batch()
        self.QFunc.training_iteration(states, ys, inds)

    def reset_game_engine(self):
        self.game_engine = SquareGameEngine(
            self.num_food, self.num_enemies, self.game_size,
            self.player_size, self.object_size
        )

    def write_summary(self, step):
        states, ys, inds = self.generate_batch()
        self.QFunc.write_summary(states, ys, inds, step)

    def save_checkpoint(self, step):
        self.QFunc.save_checkpoint(step)


DQL = DQLearning() 
DQL.epsilon = .5
DQL.display = False
def train():
    for i in range(5000):
        DQL.step()
        if i % 1000 == 0:
            DQL.reset_game_engine()

    DQL.reset_game_engine()

    for i in range(10000):
        if i % 100 == 0:
            print i
        for j in range(10):
            DQL.step()
        DQL.apply_batch()
        if i % 100 == 0:
            DQL.write_summary(DQL.steps)
        if i % 1000 == 0:
            DQL.save_checkpoint(DQL.steps)

FOLDER = '/Users/willgrathwohl/Desktop/test/'
def play_game(path, save_folder=None):
    game_size = 100
    DQL = DQLearning(checkpoint_path=path)
    DQL.display = False
    DQL.epsilon = 0.0
    r = 0
    for i in range(5000):
        im, reward, terminal = DQL.step(return_im=True)
        if save_folder is not None:
            plt.imsave(save_folder+"{}.jpg".format(i), im.astype(np.uint8))
        r += reward
        if terminal:
            return r
    return r






