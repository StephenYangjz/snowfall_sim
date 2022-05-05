import random
import math
import sys
import pygame
import snowflake
from snowflake import Snowflake
from pygame.locals import * #@UnusedWildImport

MAX_SLOPE = 45
MIN_SLOPE = -45
MIN_HEIGHT = 0

def dist_squared(P1,P2):
    return (P1[0]-P2[0])**2 + (P1[1]-P2[1])**2

class Game:
    def __init__(self, width, height, caption="Game"):
        # plethora of fields
        self.width = width
        self.height = height
        self.caption = caption
        self.framerate = 120 # FPS        
        self.foreground_color = (255, 255, 255)
        self.background_color = (21, 26, 79)
        self.snowflakes = []
        self.snowflake_counter = 0
        self.snowflake_frequency = 5
        self.snowflake_size = 3
        # self.snowflake_line = [ height - random.randint(200, 250) for i in range(width)] # for collision detection
        self.mountain_line = []
        self.mountain((0, 400), (self.width, 450), self.mountain_line)
        self.mountain_line = [int(ele[1]) for ele in self.mountain_line[:800]]
        self.snowflake_line = [e for e in self.mountain_line]
        # print(self.snowflake_line)
        # print(len(self.snowflake_line))
        self.wind_chance = 1
        self.wind_strength = 800
        self.show_text = False
        # and we're off!
        self.initialize()
        self.loop()
        
    def initialize(self):
        pygame.init()
        pygame.display.set_caption(self.caption)
        self.screen = pygame.display.set_mode((self.width, self.height))        
        self.font = pygame.font.SysFont('arial', 20)

    def loop(self):
        self.clock = pygame.time.Clock()
        while 1:
            gametime = self.clock.get_time()
            self.update(gametime)
            self.render(gametime)
            self.clock.tick(self.framerate)
    
    def update(self, gametime):
        
        # do we need to add more snow?
        if self.snowflake_counter > self.snowflake_frequency:
            self.snowflake_counter = 0
            snowflake = Snowflake(len(self.snowflakes), self.width, self.height, self.snowflake_size, self.foreground_color)
            self.snowflakes.append(snowflake)
        else:
            self.snowflake_counter += 1
            
        # what about some wind?
        w_chance = random.randint(0, 100)
        w_strength = 0
        if w_chance <= self.wind_chance:
            w_strength = random.randint(-self.wind_strength, self.wind_strength)
            
        # let it snow, let it snow, let it snow
        for snowflake in self.snowflakes:
            if snowflake.enabled:
                if w_strength != 0:
                    snowflake.wind = w_strength
                snowflake.update(gametime, self.snowflake_line)
        
        # update the other rubbish
        if self.show_text:
            self.fps_text = self.font.render("FPS: %d" % self.clock.get_fps(), 1, self.foreground_color)
            self.snowflake_text = self.font.render("Snowflakes: %d" % len(self.snowflakes), 1, self.foreground_color)
        self.handle_input(pygame.event.get())

    def mountain(self, P1,P2, result):
        if dist_squared(P1,P2) < 1:
            result.append(P2)
            return
        x1,y1 = P1
        x2,y2 = P2
        x3 = random.uniform(x1,x2)
        y3_max = min((x3-x1)*math.tan(math.radians(MAX_SLOPE)) + y1, (x2-x3)*math.tan(-math.radians(MIN_SLOPE)) + y2)
        y3_min = max((x3-x1)*math.tan(math.radians(MIN_SLOPE)) + y1, (x2-x3)*math.tan(-math.radians(MAX_SLOPE)) + y2)
        y3_min = max(y3_min, MIN_HEIGHT)
        y3 = random.uniform(y3_min,y3_max)
        P3 = (x3, y3)
        self.mountain(P1,P3, result)
        self.mountain(P3,P2, result)
        return

    def render(self, gametime):
        surface = pygame.Surface(self.screen.get_size())
        surface.convert()
        surface.fill(self.background_color)
        left_corner_x = 0
        for vertex in self.mountain_line:
            pygame.draw.line(surface, (60, 200, 60), (left_corner_x, vertex), (left_corner_x, self.height))
            left_corner_x += 1

        
        for snowflake in self.snowflakes:
            snowflake.draw(surface)
        
        if self.show_text:
            surface.blit(self.fps_text, (8, 6))
            surface.blit(self.snowflake_text, (8, 26))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
            
    def handle_input(self, events):
        for event in events:
            if event.type == pygame.QUIT: 
                sys.exit()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_d:
                    self.show_text = False if self.show_text else True 


                                      
if __name__ == "__main__": 
    game = Game(800, 600, "CS 184, Spring 2022 Final Project Deomo")
