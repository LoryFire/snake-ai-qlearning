import pygame
import random
import numpy as np

class SnakeEnv:
    def __init__(self): #costruttore
        pygame.init()
        self.width = 800
        self.height = 600
        self.block = 10
        self.speed = 15

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self): #inizializzo il gioco e lo restarto
        self.x = self.width // 2
        self.y = self.height // 2
        self.x_change = 0
        self.y_change = 0
        self.snake = [[self.x, self.y]]
        self.snake_length = 1

        self.foodx = random.randrange(0, self.width, self.block)
        self.foody = random.randrange(0, self.height, self.block)

        self.done = False
        self.score = 0
        
        self.direction = 'RIGHT'


    def step(self, action): #avanzamento del gioco
        if action == 0:
            self.x_change = -self.block
            self.y_change = 0
        elif action == 1:
            self.x_change = self.block
            self.y_change = 0
        elif action == 2:
            self.x_change = 0
            self.y_change = -self.block
        elif action == 3:
            self.x_change = 0
            self.y_change = self.block

        self.x += self.x_change
        self.y += self.y_change
        new_head = [self.x, self.y]
        self.snake.append(new_head)

        if len(self.snake) > self.snake_length:
            self.snake.pop(0)

        if (
            self.x < 0 or self.x >= self.width or
            self.y < 0 or self.y >= self.height or
            new_head in self.snake[:-1]
        ):
            self.done = True
            return -10  # ricompensa negativa per morte

        reward = 0
        if self.x == self.foodx and self.y == self.foody:
            self.snake_length += 1
            self.score += 1
            reward = 10 #ricompensa positiva per mangiare
            self.foodx = random.randrange(0, self.width, self.block)
            self.foody = random.randrange(0, self.height, self.block)

        return reward

    def render(self): #aggiorno graficamente il gioco
        self.display.fill(self.blue)
        pygame.draw.rect(self.display, self.green, [self.foodx, self.foody, self.block, self.block])
        for part in self.snake:
            pygame.draw.rect(self.display, self.black, [part[0], part[1], self.block, self.block])
        pygame.display.update()
        self.clock.tick(self.speed)

#AGGIUNTA metodo per verificare le collisioni

    def _is_collision(self, point):
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if point in self.snake[1:]:
            return True
        return False

    '''
        def get_state(self):
            head = self.snake[-1]
            dx = self.foodx - head[0]
            dy = self.foody - head[1]
            return [head[0], head[1], dx, dy]
    '''
#AGGIORNAMENTO DELLO STATO: non più posizione della testa e del cibo

'''def get_state(self):
    head = self.snake[0]
    point_l = [head[0] - self.block_size, head[1]]
    point_r = [head[0] + self.block_size, head[1]]
    point_u = [head[0], head[1] - self.block_size]
    point_d = [head[0], head[1] + self.block_size]

    dir_l = self.direction == 'LEFT'
    dir_r = self.direction == 'RIGHT'
    dir_u = self.direction == 'UP'
    dir_d = self.direction == 'DOWN'

    danger_straight = (
        (dir_r and self._is_collision(point_r)) or
        (dir_l and self._is_collision(point_l)) or
        (dir_u and self._is_collision(point_u)) or
        (dir_d and self._is_collision(point_d))
    )

    danger_right = (
        (dir_u and self._is_collision(point_r)) or
        (dir_d and self._is_collision(point_l)) or
        (dir_l and self._is_collision(point_u)) or
        (dir_r and self._is_collision(point_d))
    )

    danger_left = (
        (dir_d and self._is_collision(point_r)) or
        (dir_u and self._is_collision(point_l)) or
        (dir_r and self._is_collision(point_u)) or
        (dir_l and self._is_collision(point_d))
    )

    food_left = self.food[0] < head[0]
    food_right = self.food[0] > head[0]
    food_up = self.food[1] < head[1]
    food_down = self.food[1] > head[1]

    state = [
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dir_l),
        int(dir_r),
        int(dir_u),
        int(dir_d),
        int(food_left),
        int(food_right),
        int(food_up),
        int(food_down)
    ]

    return np.array(state, dtype=int)'''

#AGGIORNAMENTO DELLO STATO: restituisce 11 vettori e non coordinate

def get_state(self):
    head = self.snake[0]
    point_l = [head[0] - self.block_size, head[1]]
    point_r = [head[0] + self.block_size, head[1]]
    point_u = [head[0], head[1] - self.block_size]
    point_d = [head[0], head[1] + self.block_size]

    dir_l = self.direction == 'LEFT'
    dir_r = self.direction == 'RIGHT'
    dir_u = self.direction == 'UP'
    dir_d = self.direction == 'DOWN'

    danger_straight = (
        (dir_r and self._is_collision(point_r)) or
        (dir_l and self._is_collision(point_l)) or
        (dir_u and self._is_collision(point_u)) or
        (dir_d and self._is_collision(point_d))
    )

    danger_right = (
        (dir_u and self._is_collision(point_r)) or
        (dir_d and self._is_collision(point_l)) or
        (dir_l and self._is_collision(point_u)) or
        (dir_r and self._is_collision(point_d))
    )

    danger_left = (
        (dir_d and self._is_collision(point_r)) or
        (dir_u and self._is_collision(point_l)) or
        (dir_r and self._is_collision(point_u)) or
        (dir_l and self._is_collision(point_d))
    )

    food_left = self.food[0] < head[0]
    food_right = self.food[0] > head[0]
    food_up = self.food[1] < head[1]
    food_down = self.food[1] > head[1]

    state = [
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dir_l),
        int(dir_r),
        int(dir_u),
        int(dir_d),
        int(food_left),
        int(food_right),
        int(food_up),
        int(food_down)
    ]

    return np.array(state, dtype=int)


    



