import time as t
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

class ball:
    def __init__(self,framesize, x=0, y=0, color=255, direction=[1,0],obstacle_color=200,destructible_obstacles=False,boundary_color=50,player_color=255):
        self.x = x
        self.y = y
        self.direction = direction
        self.color = color
        self.obstacle_color = obstacle_color
        self.boundary_color = boundary_color
        self.destructible_obstacles = destructible_obstacles
        self.framesize=framesize
        self.frames = np.zeros(framesize, np.uint8)
        self.frames[self.x,self.y]=self.color
        self.frame_nr=0

        self.player_direction = [0,0]
        self.player_position = [int(framesize[0]/2),int(framesize[1]/10)+2]
        self.player_width = int(framesize[0]/10)
        self.player_color = player_color

        self.obstacle_space = []
        self.boundary_space = []
        self.player_space = []
        self.bounced_blocks = []
        self.populate_restricted_space()
        for coor in self.restricted_space:
            self.frames[coor[0],coor[1]]=self.boundary_color
        
        
    def populate_restricted_space(self):
        self.restricted_space = []
        for x in range(self.framesize[0]):
            for y in range(self.framesize[1]):
                if x<=0 or y<=0 or x>=self.framesize[0]-1 or y>=self.framesize[1]-1:
                    self.restricted_space.append([x,y])
                    self.boundary_space.append([x,y])
    def add_obstacles(self,obstacles):
        for obstacle in obstacles:
            self.restricted_space.append(obstacle)
            self.obstacle_space.append(obstacle)
    def crossing_x_border(self,x1):
        if [x1,self.y] in self.restricted_space:
            self.bounced_blocks.append([x1,self.y])
            return True
        elif [x1,self.y] in self.player_space:
            self.bounced_blocks.append([x1,self.y])
            return True
        else:
            return False
    def crossing_y_border(self,y1):
        if [self.x,y1] in self.restricted_space:
            self.bounced_blocks.append([self.x,y1])
            return True
        elif [self.x,y1] in self.player_space:
            self.bounced_blocks.append([self.x,y1])
            self.direction[0] = self.direction[0]-self.player_direction[0]

            if self.direction[0]>1:#angle the ball
                self.direction[0] = 1
            if self.direction[0]<-1:
                self.direction[0] = -1
            return True
        else:
            return False
    def crossing_xy_border(self,x1,y1):
        if [x1,y1] in self.restricted_space:
            self.bounced_blocks.append([x1,y1])
            return True
        elif [x1,y1] in self.player_space:
            self.bounced_blocks.append([x1,y1])
            return True
        else:
            return False
        
    def bounce(self,x1,y1):
        x_blocked=False
        y_blocked=False
        self.bounced_blocks=[]
        if self.crossing_x_border(x1):
            self.direction[0]=-self.direction[0]
            x_blocked=True
        if self.crossing_y_border(y1):
            self.direction[1]=-self.direction[1]
            y_blocked=True
        if not x_blocked and not y_blocked and self.crossing_xy_border(x1,y1):
            self.direction[0]=-self.direction[0]
            self.direction[1]=-self.direction[1]
        if self.destructible_obstacles:
            self.destroy_blocks()

    def destroy_blocks(self):
        for block in self.bounced_blocks:
            if block in self.obstacle_space:
                self.obstacle_space.remove(block)
                self.restricted_space.remove(block)
    def move_ball(self):
        x1,y1 = self.x+self.direction[0],self.y+self.direction[1]
        if any([self.crossing_x_border(x1), self.crossing_y_border(y1),self.crossing_xy_border(x1,y1)]):
            self.bounce(x1,y1)
        self.x,self.y = self.x+self.direction[0],self.y+self.direction[1]
        #self.refresh_frame()
            
    def draw_frame(self):
        frame = np.zeros(self.framesize, np.uint8)
        for coor in self.restricted_space:
            if coor in self.obstacle_space:
                frame[coor[0],coor[1]]=self.obstacle_color
            elif coor in self.boundary_space:
                frame[coor[0],coor[1]]=self.boundary_color
        for coor in self.player_space:
            frame[coor[0],coor[1]]=self.player_color
        #x1,y1 = self.x+self.direction[0],self.y+self.direction[1]
        x1,y1 = self.x,self.y
        frame[x1,y1]=self.color
        #self.x = x1
        #self.y = y1
        return frame

    def refresh_frame(self):
        self.frame_nr+=1
        if self.frame_nr%2==0:
            self.move_ball()
        self.move_player()
        frame=self.draw_frame()
        return frame
        #self.frames = np.dstack((self.frames,frame))
    
    def player_out_of_bounds(self,direction):
        x1,y1 = self.player_position[0]+direction[0],self.player_position[1]+direction[1]
        return x1+self.player_width>self.framesize[0] or x1<0
    def move_player(self):
        #print("moving player",self.player_direction,self.player_position,self.player_space)
        if not self.player_out_of_bounds(self.player_direction):
            self.player_position=[self.player_direction[0]+self.player_position[0],self.player_direction[1]+self.player_position[1]]
        self.player_space=[]
        for x in range(self.player_width):
            self.player_space.append([self.player_position[0]+x,self.player_position[1]])
    def on_key_press(self,event):
        print("key pressed",event.key)
        if event.key == 'right':
            self.player_direction = [1,0]
        elif event.key == 'left':
            self.player_direction = [-1,0]
        else:
            self.player_direction = [0,0]
    def on_key_release(self,event):
        print("key released",event.key)
        if event.key == 'right' or event.key == 'left':
            self.player_direction = [0,0]

    def main_game_loop(self):
        plt.ion()
        fig = plt.figure(1)
        while True:
            #self.player_direction=[0,0]
            fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            fig.canvas.mpl_connect('key_release_event', self.on_key_release)
            frame=self.refresh_frame()
            #frameT = frame.transpose((1,0))
            #plt.clf()
            plt.imshow(np.flip(frame.T,0))
            plt.pause(0.03)
            plt.clf()



    def get_action_space(self):
        return [0, 1, 2]
    def action_to_direction(self,action):
        if action==0:
            return [1,0]
        elif action==1:
            return [-1,0]
        else:
            return [0,0]
    def ball_direction_to_int(self,direction):
        if direction==[0,1]:
            return 0
        elif direction==[0,-1]:
            return 1
        elif direction==[1,1]:
            return 2
        elif direction==[1,-1]:
            return 3
        elif direction==[-1,1]:
            return 4
        elif direction==[-1,-1]:
            return 5
    def player_direction_to_int(self,direction):
        return direction[0]+1

        
    def main_game_loop_no_visuals(self):
        #3 actions: left,right,none
        #states: x,y,ball_directions,playerx,player_directions
        epsilon = 1
        alpha=0.1
        gamma=0.99
        state = (16,16,3,16,0)
        nr_states = self.framesize[0]*self.framesize[1]*6*self.framesize[0]*3
        q_table=np.zeros((self.framesize[0],self.framesize[1],6,self.framesize[0],3,3))
        q_table=np.random.rand(self.framesize[0],self.framesize[1],6,self.framesize[0],3,3)
        for i in range(10000000):
            done = False

            #while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(self.get_action_space())
            else:
                action = np.argmax(q_table[state])

            self.player_direction=self.action_to_direction(action)

            self.refresh_frame()#move player and ball
            next_state=(self.x,self.y,self.ball_direction_to_int(self.direction),self.player_position[0],self.player_direction_to_int(self.player_direction))
            reward = len(self.bounced_blocks)
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])
            q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
            epsilon*=0.99

            self.q_table=q_table
        return q_table
            
            
    def toTensor(self):
        self.frame_tensor = torch.from_numpy(self.frames)
        self.frame_tensor = self.frame_tensor.permute((2,0,1))
        return self.frame_tensor

def create_frame_batch(nr_frames,batch_size=32,framesize=(32,32),color=255,noise=0,boundary_shot_probability=0.5,nr_obstacles=0,obstacle_color=255,destructible_obstacles=False):
    batch=torch.empty((batch_size,nr_frames,framesize[0],framesize[1]),dtype=torch.float32)

    for i in range(batch_size):
        if random.random()<boundary_shot_probability:
            if random.random()<0.5:
                if random.random()<0.5:
                    x=random.randint(1,3)
                else:
                    x=random.randint(framesize[0]-4,framesize[0]-2)
            else:
                x=random.randint(1,framesize[0]-2)
            if random.random()<0.5:
                if random.random()<0.5:
                    y=random.randint(1,3)
                else:
                    y=random.randint(framesize[1]-4,framesize[1]-2)
            else:
                y=random.randint(1,framesize[1]-2)
        else:
            x,y=random.randint(1,framesize[0]-2),random.randint(1,framesize[1]-2)
        direction = [random.randint(-1,1),random.randint(-1,1)]
        if direction[0]==0 and direction[1]==0:
            direction = [1,0]
        b=ball(framesize=framesize,x=x,y=y,color=color,direction=direction,obstacle_color=obstacle_color,destructible_obstacles=destructible_obstacles)
        for j in range(nr_obstacles):
            xo,yo=random.randint(4,framesize[0]-5),random.randint(4,framesize[1]-5)
            if xo!=x and yo!=y:
                b.add_obstacles([[xo,yo]])
        for j in range(nr_frames-1):
            b.move_ball()
        instance=b.toTensor()
        batch[i]=instance
    return batch
