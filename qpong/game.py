import pygame
import numpy as np
import random
import math

class Pong:
    def __init__(self, max_score=100, render=False):

        self.scale = 2
        self.height = 600 * self.scale
        self.width = 600 * self.scale
        
        pygame.init()
        self.render = render
        
        if self.render:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Q-Pong')
        self.clock = pygame.time.Clock()



        self.paddle_width = 8 * self.scale
        self.paddle_height = 80 * self.scale
        self.paddle_speed = 8 * self.scale

        self.ball_size = 20 * self.scale
        self.ball_speed = 5 * self.scale

        self.left_padding = 0 * self.scale
        self.right_padding = self.width - self.paddle_width - self.left_padding

        # rect (left, top, width, height)
        self.left_paddle = pygame.Rect(
            self.left_padding,
            self.height // 2 - self.paddle_height // 2,
            self.paddle_width,
            self.paddle_height
        )

        self.right_paddle = pygame.Rect(
            self.right_padding,
            self.height // 2 - self.paddle_height // 2,
            self.paddle_width,
            self.paddle_height
        )

        self.ball = pygame.Rect(
            self.width // 2 - self.ball_size // 2,
            self.height // 2 - self.ball_size // 2,
            self.ball_size,
            self.ball_size
        )

        # state
        self.place_ball()

        self.left_score = 0
        self.right_score = 0
        self.max_score = max_score

    def place_ball(self):
        self.ball.center = (self.width // 2, self.height // 2)
        angle = random.uniform(-math.pi / 4, math.pi / 4)  # Random angle between -45° and 45°
        direction = random.choice([-1, 1])  # Left or right launch
        self.ball_speed_x = self.ball_speed * math.cos(angle) * direction
        self.ball_speed_y = self.ball_speed * math.sin(angle)

    def get_state(self):
        return np.array([
            self.left_paddle.centery / self.height,
            self.right_paddle.centery / self.height,
            self.ball.centerx / self.width,
            self.ball.centery / self.height,
            self.ball_speed_x / self.ball_speed,
            self.ball_speed_y / self.ball_speed
        ])

    def step(self, left_action=0, right_action=0):
        """
        action space: [-1, 0, 1] :: [up, none, down]
        """

        reward = 0

        # left paddle movement
        if left_action == -1 and self.left_paddle.top > 0:
            self.left_paddle.y -= self.paddle_speed
        elif left_action == 1 and self.left_paddle.bottom < self.height:
            self.left_paddle.y += self.paddle_speed

        # right paddle movement
        if right_action == -1 and self.right_paddle.top > 0:
            self.right_paddle.y -= self.paddle_speed
        elif right_action == 1 and self.right_paddle.bottom < self.height:
            self.right_paddle.y += self.paddle_speed

        # move ball
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # ball hits vertical border
        if self.ball.top <= 0 or self.ball.bottom >= self.height:
            self.ball_speed_y *= -1 # bounce back along Yaxis

        # ball collides with paddles
        if self.ball.colliderect(self.left_paddle) or self.ball.colliderect(self.right_paddle):
            angle = random.uniform(-15, 15)  # Random variation in bounce angle (-30° to 30°)
            speed = (self.ball_speed_x ** 2 + self.ball_speed_y ** 2) ** 0.5  # Maintain speed magnitude
            
            self.ball_speed_x = -self.ball_speed_x  # Reverse direction
            self.ball_speed_y += math.tan(math.radians(angle)) * abs(self.ball_speed_x)  # Adjust Y direction

            speed_factor = speed / ((self.ball_speed_x ** 2 + self.ball_speed_y ** 2) ** 0.5)
            self.ball_speed_x *= speed_factor
            self.ball_speed_y *= speed_factor
            reward = 1

        # ball goes beyond either paddle
        if self.ball.left < 0:
            self.right_score += 1
            reward = -1
            self.place_ball()

        if self.ball.right > self.width:
            self.left_score += 1
            reward = -1
            self.place_ball()

        done = self.left_score == 10 or self.right_score == 10

        return self.get_state(), reward, done, (self.left_score, self.right_score)
    
    def draw_dashed_line(self, surface, color, start_pos, end_pos, width=5, dash_length=15, space_length=10):
        x1, y1 = start_pos
        x2, y2 = end_pos
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Line length
        dx, dy = (x2 - x1) / length, (y2 - y1) / length  # Unit vector

        num_dashes = int(length // (dash_length + space_length))
        
        for i in range(num_dashes):
            start_x = x1 + (dash_length + space_length) * i * dx
            start_y = y1 + (dash_length + space_length) * i * dy
            end_x = start_x + dash_length * dx
            end_y = start_y + dash_length * dy
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)

    
    def show(self,):

        self.screen.fill('black')
        
        self.draw_dashed_line(
            self.screen,
            (150, 150, 150),  # Gray color
            (self.width // 2, 0),
            (self.width // 2, self.height),
            width=6,  # Thickness
            dash_length=20,
            space_length=10
        )
        
        # Draw paddles and ball
        pygame.draw.rect(self.screen, 'white', self.left_paddle)
        pygame.draw.rect(self.screen, 'white', self.right_paddle)
        pygame.draw.ellipse(self.screen, 'white', self.ball)
        
        # Draw scores
        font = pygame.font.Font(None, 36 * self.scale)
        left_text = font.render(str(self.left_score), True, 'white')
        right_text = font.render(str(self.right_score), True, 'white')
        self.screen.blit(left_text, (self.width//4, 20 * self.scale))
        self.screen.blit(right_text, (3*self.width//4, 20 * self.scale))
        
        pygame.display.flip()


if __name__ == '__main__':
    
    game = Pong(render = True)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        left_action = 0
        right_action = 0
        
        if keys[pygame.K_w]: left_action = -1
        if keys[pygame.K_s]: left_action = 1
        if keys[pygame.K_UP]: right_action = -1
        if keys[pygame.K_DOWN]: right_action = 1
        
        # Update game
        state, reward, done, score = game.step(left_action, right_action)
        if game.render:
            game.show()
        game.clock.tick(60)
        print(game.get_state())
        
        if done:
            running = False

    pygame.quit()