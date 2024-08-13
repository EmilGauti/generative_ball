import time as t
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

class ball():
    def __init__(self,x=2, y=2,direction=[1,0], color=255):
        self.x = x
        self.y = y
        self.color = color
        self.direction = direction
        speed = np.sqrt(self.direction[0]**2+self.direction[1]**2)
        self.direction[0] = self.direction[0]/speed
        self.direction[1] = self.direction[1]/speed
        self.player_bounced=False
class GameSpace():
    def __init__(self,framesize=(32,32), xy=[[2,2]], direction=[[1,0]],destructible_obstacles=True,nr_sim_steps=10,nr_balls=1):
        self.has_ended=False
        self.balls = [ball(x=xy[i][0],y=xy[i][1],direction=direction[i]) for i in range(nr_balls)]
        
        self.destructible_obstacles = destructible_obstacles
        self.framesize=framesize
        self.sim_steps=nr_sim_steps

        self.player_direction = [0,0]
        self.player_position = [int(framesize[0]/2),int(framesize[1]/10)+2]
        self.player_width = int(framesize[0]/10)


        self.obstacle_space = []
        self.boundary_space = []
        self.terminating_boundary_space = []
        self.player_space = []
        self.bounced_blocks = []
        self.populate_restricted_space()

    def get_state(self):
        current_state=[]
        for b in self.balls:
            current_state.extend([b.x,b.y,b.direction[0],b.direction[1]])
        current_state.append(self.player_position[0])
        return current_state
        
        
    def populate_restricted_space(self):
        self.restricted_space = []
        for x in range(self.framesize[0]):
            for y in range(self.framesize[1]):
                if x<=0 or y<=0 or x>=self.framesize[0]-1 or y>=self.framesize[1]-1:
                    self.restricted_space.append([x,y])
                    self.boundary_space.append([x,y])
                if y<=0:
                    self.terminating_boundary_space.append([x,y])
    def add_obstacles(self,obstacles):
        for obstacle in obstacles:
            self.restricted_space.append(obstacle)
            self.obstacle_space.append(obstacle)
    def crossing_x_border(self,x1,y0,b):
        x1,y0=int(x1),int(y0)
        if [x1,y0] in self.restricted_space:
            self.bounced_blocks.append([x1,y0])
            return True
        elif [x1,y0] in self.player_space:
            self.bounced_blocks.append([x1,y0])
            b.player_bounced = True
            return True
        else:
            return False
    def crossing_y_border(self,x0,y1,b):
        x0,y1=int(x0),int(y1)
        if [x0,y1] in self.restricted_space:
            self.bounced_blocks.append([x0,y1])
            return True
        elif [x0,y1] in self.player_space:
            self.bounced_blocks.append([x0,y1])
            b.direction[0] = b.direction[0]+0.5*self.player_direction[0]
            speed = np.sqrt(b.direction[0]**2+b.direction[1]**2)
            b.direction[0] = b.direction[0]/speed
            b.direction[1] = b.direction[1]/speed
            b.player_bounced = True

            return True
        else:
            return False
    def crossing_xy_border(self,x1,y1,b):
        x1,y1=int(x1),int(y1)
        if [x1,y1] in self.restricted_space:
            self.bounced_blocks.append([x1,y1])
            return True
        elif [x1,y1] in self.player_space:
            self.bounced_blocks.append([x1,y1])
            b.player_bounced = True
            return True
        else:
            return False
        
    def bounce(self,x1,y1,b):
        self.bounced_blocks=[]
        x_blocked=False
        y_blocked=False
        
        if self.crossing_x_border(x1,b.y,b):
            b.direction[0]=-b.direction[0]
            x_blocked=True
        if self.crossing_y_border(b.x,y1,b):
            b.direction[1]=-b.direction[1]
            y_blocked=True
        if not x_blocked and not y_blocked and self.crossing_xy_border(x1,y1,b):
            b.direction[0]=-b.direction[0]
            b.direction[1]=-b.direction[1]
        if self.destructible_obstacles:
            self.destroy_blocks()

    def destroy_blocks(self):
        for block in self.bounced_blocks:
            if block in self.obstacle_space:
                self.obstacle_space.remove(block)
                self.restricted_space.remove(block)
    def move_ball(self):
        for b in self.balls:
            b.player_bounced=False
        for j in range(self.sim_steps):
            for i,b in enumerate(self.balls):
                x1,y1 = b.x+b.direction[0]/self.sim_steps,b.y+b.direction[1]/self.sim_steps
                if self.ball_out_of_bounds(x1,y1):
                    self.balls.remove(b)
                if any([self.crossing_x_border(x1,b.y,b), self.crossing_y_border(b.x,y1,b),self.crossing_xy_border(x1,y1,b)]):
                    self.bounce(x1,y1,b)
               

                b.x,b.y = b.x+b.direction[0]/self.sim_steps,b.y+b.direction[1]/self.sim_steps
        if len(self.balls)==0:
            self.has_ended=True
    def ball_out_of_bounds(self,x1,y1):
        x1,y1=int(x1),int(y1)
        if [x1,y1] in self.terminating_boundary_space:
            return True
        return False


        #self.refresh_frame()
    def player_out_of_bounds(self,direction):
        x1,y1 = self.player_position[0]+direction[0],self.player_position[1]+direction[1]
        return x1+self.player_width>self.framesize[0] or x1<0
    def move_player(self):
        #print("moving player",self.player_direction,self.player_position,self.player_space)
        if not self.player_out_of_bounds(self.player_direction):
            #print("pos was",self.player_position)
            self.player_position=[self.player_direction[0]+self.player_position[0],self.player_direction[1]+self.player_position[1]]
            #print("pos is",self.player_position)
        self.player_space=[]
        for x in range(self.player_width):
            self.player_space.append([self.player_position[0]+x,self.player_position[1]])
    def on_key_press(self,event):
        print("key pressed",event.key)
        if event.key == 'right':
            self.player_direction = [2,0]
        elif event.key == 'left':
            self.player_direction = [-2,0]
        else:
            self.player_direction = [0,0]
    def on_key_release(self,event):
        print("key released",event.key)
        if event.key == 'right' or event.key == 'left':
            self.player_direction = [0,0]

class DrawSpace():

    def __init__(self,color=255,player_color=255,obstacle_color=200,boundary_color=50):

        self.color = color
        self.player_color = player_color
        self.obstacle_color = obstacle_color
        self.boundary_color = boundary_color

    def refresh_frame(self,gs):
        self.frames = np.zeros(gs.framesize, np.uint8)
        for coor in gs.restricted_space:
            if coor in gs.obstacle_space:
                self.frames[coor[0],coor[1]]=self.obstacle_color
            elif coor in gs.boundary_space:
                self.frames[coor[0],coor[1]]=self.boundary_color
        for coor in gs.player_space:
            self.frames[coor[0],coor[1]]=self.player_color
        #x1,y1 = self.x+self.direction[0],self.y+self.direction[1]
        for b in gs.balls:
            x1,y1 = int(b.x),int(b.y)
            self.frames[x1,y1]=b.color

class BreakoutGame():
    def __init__(self,framesize, xy=[[2,2]], color=255, direction=[[1,0]],nr_sim_steps=10,obstacle_color=200,destructible_obstacles=False,boundary_color=50,player_color=255,nr_balls=1):
        self.gs = GameSpace(framesize, xy, direction,destructible_obstacles,nr_sim_steps,nr_balls)
        self.ds = DrawSpace(color,player_color,obstacle_color,boundary_color)
        self.has_ended = self.gs.has_ended

    def refresh_frame(self):
        self.gs.move_ball()
        self.gs.move_player()
        self.ds.refresh_frame(self.gs)
        self.has_ended = self.gs.has_ended
        return self.ds.frames
    def main_game_loop(self):
        plt.ion()
        fig = plt.figure(1)
        while not self.has_ended:
            #self.player_direction=[0,0]
            fig.canvas.mpl_connect('key_press_event', self.gs.on_key_press)
            fig.canvas.mpl_connect('key_release_event', self.gs.on_key_release)
            frame=self.refresh_frame()
            #frameT = frame.transpose((1,0))
            #plt.clf()
            plt.imshow(np.flip(frame.T,0))
            plt.pause(0.03)
            plt.clf()

    def rl_refresh_frame(self,action):
        self.gs.move_ball()
        self.gs.player_direction=action
        self.gs.move_player()
        self.has_ended = self.gs.has_ended
        next_state = self.gs.get_state()
        reward=0
        try:
            reward += 1/(abs(self.gs.player_position[0]-self.gs.balls[0].x)+0.01)
            print(reward,abs(self.gs.player_position[0]-self.gs.balls[0].x))
        except:
            reward=0
        for i in range(len(self.gs.balls)):
            reward+=self.gs.balls[i].player_bounced

        return next_state,reward,self.has_ended
        
