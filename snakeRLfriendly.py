import pygame
import random

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

#AGGIORNAMENTO DELLO STATO: non più posizione della testa e del cibo

    '''
        def get_state(self):
            head = self.snake[-1]
            dx = self.foodx - head[0]
            dy = self.foody - head[1]
            return [head[0], head[1], dx, dy]
    '''
    
    def get_state(self): # restituisce lo stato con 0 e 1: pericolo, direzione, posizione cibo
        head = self.snake[-1]
        x_change = self.x_change
        y_change = self.y_change
        block = self.block

        # punti in ogni direzione
        point_l = [head[0] - block, head[1]]
        point_r = [head[0] + block, head[1]]
        point_u = [head[0], head[1] - block]
        point_d = [head[0], head[1] + block]

        # posizione corrente
        dir_l = x_change == -block
        dir_r = x_change == block
        dir_u = y_change == -block
        dir_d = y_change == block

        def danger(point):
            return (
                point[0] >= self.width or point[0] < 0 or
                point[1] >= self.height or point[1] < 0 or
                point in self.snake
            )

        state = [
            # pericolo
            danger([head[0] + x_change, head[1] + y_change]),
            danger([head[0] + y_change, head[1] - x_change]),  # destra
            danger([head[0] - y_change, head[1] + x_change]),  # sinistra

            # direzioni
            dir_l, dir_r, dir_u, dir_d,

            # posizione del cibo
            self.foodx < head[0],  # sinistra
            self.foodx > head[0],  # destra
            self.foody < head[1],  # sopra
            self.foody > head[1],  # sotto
        ]

        return list(map(int, state))  # converti True False in 1 0
    panino


    



