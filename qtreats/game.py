import pygame
import random

class TreatsEnv:
    def __init__(self, grid_size=4, tile_size=256, render=False):
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.render = render
    
        pygame.init()

        # Load and scale images
        self.images = {
            'player': pygame.transform.scale(pygame.image.load("assets/dog.png"), (tile_size, tile_size)),
            'F': pygame.transform.scale(pygame.image.load("assets/normal.png"), (tile_size, tile_size)),
            'H': pygame.transform.scale(pygame.image.load("assets/enemy.png"), (tile_size, tile_size)),
            'G': pygame.transform.scale(pygame.image.load("assets/goal.png"), (tile_size, tile_size)),
            'woof': pygame.transform.scale(pygame.image.load("assets/woof.png"), (tile_size * grid_size, tile_size * grid_size)),
            'meow': pygame.transform.scale(pygame.image.load("assets/meow.png"), (tile_size * grid_size, tile_size * grid_size))
        }

        # Map layout (S=Start, H=Hole, G=Goal, F=Floor)
        self.map = [
            ["S", "F", "F", "F"],
            ["F", "H", "F", "H"],
            ["F", "F", "F", "H"],
            ["H", "F", "F", "G"]
        ]

        # Action mapping (left, down, right, up)
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.width = self.grid_size
        self.height = self.grid_size

        self.reset()

    def reset(self):
        self.player_x, self.player_y = random.randint(0,2), 0
        self.game_over = False

        if self.render:
            self.screen = pygame.display.set_mode((self.grid_size * self.tile_size, self.grid_size * self.tile_size))
            pygame.display.set_caption("Treats")

        self.clock = pygame.time.Clock()

        return self.get_state()

    def get_state(self):
        return self.player_y * self.width + self.player_x

    def step(self, action):
        dx, dy = self.actions[action]
        new_x = self.player_x + dx
        new_y = self.player_y + dy

        # Check boundaries
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.player_x, self.player_y = new_x, new_y

        reward = 0
        done = False

        current_tile = self.map[self.player_y][self.player_x]
        if current_tile == "G":
            reward = 1
            done = True
        elif current_tile == "H":
            done = True

        if self.render:
            self.clock.tick(1)
        else:
            self.clock.tick()

        return self.get_state(), reward, done

    def show(self):
        if self.render:
            self.screen.fill((255, 255, 255))

            # Draw grid
            for y in range(self.height):
                for x in range(self.width):
                    tile = self.map[y][x]
                    image = self.images.get(tile, self.images['F'])
                    self.screen.blit(image, (x * self.tile_size, y * self.tile_size))

            # Draw player
            self.screen.blit(self.images['player'], (self.player_x * self.tile_size, self.player_y * self.tile_size))
            
            decision_tile = self.map[self.player_y][self.player_x] 
            if decision_tile == 'H':
                self.screen.blit(self.images['meow'], (0,0))
            if decision_tile == 'G':
                self.screen.blit(self.images['woof'], (0,0))
            
            pygame.display.flip()

if __name__ == "__main__":
    env = TreatsEnv(render=False)
    running = True
    
    while running:
        state = env.reset()
        done = False
        
        while not done and running:
            env.show()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if running:
                action = random.randint(0, 3)
                next_state, reward, done = env.step(action)
                print(action, next_state, reward)
        
        if done and running:
            env.show()
            if env.render:
                pygame.time.wait(1000)
    
    pygame.quit()