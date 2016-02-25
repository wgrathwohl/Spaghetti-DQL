"""
File for main game engine. Will be a simple game where you are a
square and you need to move on top of black enemies and avoid red
enimies
"""
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

# stores move name and update values
MOVES = {
    "none":  (0,  0),
    "up":    (-1, 0),
    "down":  (1,  0),
    "left":  (0, -1),
    "right": (0,  1)
}
COLOR_INDS = {
    "red": 0,
    "green": 1,
    "blue": 2
}

MOVE_INDS = [
    "none",
    "up",
    "down",
    "left",
    "right",
]

def random_loc(game_size):
    return (np.random.randint(0, game_size), np.random.randint(0, game_size))

def update_loc(loc, update, size):
    x = loc[0] + update[0]
    y = loc[1] + update[1]
    x = max(min(x, size - 2), 0)
    y = max(min(y, size - 2), 0)
    return x, y

def corners(loc, size):
    c0 = loc
    c1 = loc[0] + size - 1, loc[1]
    c2 = loc[0], loc[1] + size - 1
    c3 = loc[0] + size - 1, loc[1] + size - 1
    return c0, c1, c2, c3

def isinside(corner, obj):
    if corner[0] >= obj.loc[0] \
    and corner[1] >= obj.loc[1] \
    and corner[0] <= obj.loc[0] + obj.size - 1 \
    and corner[1] <= obj.loc[1] + obj.size - 1:
        return True
    return False

def overlap(o1, o2):
    """
    Returns true if o1 and o2 overlap
    o1 and o2 must have .size and .loc methods
    """
    return any([isinside(corner, o2) for corner in corners(o1.loc, o1.size)])



class MovingObject:
    def __init__(self, game_size, size, color, stationary_prob=.5):
        self.game_size = game_size
        self.size = size
        self.loc = random_loc(game_size)
        self.color = color
        self.stationary_prob = stationary_prob
        self.dead = False
    def move(self):
        if np.random.rand() > self.stationary_prob:
            move_ind = np.random.randint(0, len(MOVES))
            move = MOVES.values()[move_ind]
            self.loc = update_loc(self.loc, move, self.game_size)


class Player:
    def __init__(self, game_size, size, color):
        self.game_size = game_size
        self.size = size
        self.color = color
        self.loc = (game_size / 2, game_size / 2)
        self.dead = False
    def move(self, action):
        assert action in MOVES, "illegal action {}".format(action)
        self.loc = update_loc(self.loc, MOVES[action], self.game_size)


class SquareGameEngine:
    def __init__(self, num_food, num_enemies, game_size=100, player_size=4, object_size=2):
        self.num_food = num_food
        self.num_enemies = num_enemies
        self.game_size = game_size
        self.player_size = player_size

        self.foods = [MovingObject(game_size, object_size, "green") for i in range(self.num_food)]
        self.enemies = [MovingObject(game_size, object_size, "red") for i in range(self.num_enemies)]
        self.player = Player(self.game_size, self.player_size, "blue")
        self.dead_food_cnt = 0
        self.dead_enemy_cnt = 0

    def update(self, action):
        self.player.move(action)
        reward = 0
        for food in self.foods:
            if not food.dead:
                food.move()
                if overlap(self.player, food):
                    food.dead = True
                    self.dead_food_cnt += 1
                    reward += 1
        for enemy in self.enemies:
            if not enemy.dead:
                enemy.move()
                if overlap(self.player, enemy):
                    enemy.dead = True
                    self.dead_enemy_cnt += 1
                    reward -= 1
        if self.dead_food_cnt == self.num_food:
            return reward, True
        else:
            return reward, False

    def render(self):
        im = np.zeros((self.game_size, self.game_size, 3), dtype=np.int32)
        for obj in self.foods + self.enemies + [self.player]:
            if not obj.dead:
                l = obj.loc
                s = obj.size
                c = obj.color
                im[l[0]: l[0] + s, l[1]: l[1] + s, COLOR_INDS[c]] = 255
        return im
    

def play_game():
    game_size = 100
    game = SquareGameEngine(10, 10, game_size, 4, 2)
    pygame.init()
    screen = pygame.display.set_mode((game_size, game_size))
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 'up'
        elif keys[pygame.K_DOWN]:
            action = 'down'
        elif keys[pygame.K_LEFT]:
            action = 'left'
        elif keys[pygame.K_RIGHT]:
            action = 'right'
        else:
            action = 'none'
            #game.update(action)
        r, terminal = game.update(action)
        im = game.render()

        #numpy_surface = np.frombuffer(surface.get_buffer())
        #numpy_surface[...] = np.frombuffer(im)
        #del numpy_surface
        surface = pygame.surfarray.make_surface(np.swapaxes(im, 0, 1))

        screen.blit(surface,(0, 0))
        pygame.display.flip()
        print r
        #plt.imshow(im)
        #plt.show()
        if terminal:
            print "RESTART"
            game = SquareGameEngine(10, 10, game_size, 4, 2)
        time.sleep(0.1)
        pygame.event.pump()




