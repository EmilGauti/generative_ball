# Summary
This is a fun hobby project to play with frame prediction and reinforcement learning for the video game of Breakout.

In this project I develop an agent using Deep-Q-Network(DQN) that can play a custom made version of breakout. In addition I develop a U-Net Convolutional Neural Network (CNN) that reads three sequential frames and predicts the fourth.

![model](generative_ball_model.drawio.png)

Below is a demo of an early version of the frame prediction. It fails in breaking the blocks when hitting from the corners but works as expected when bouncing of walls and breaks blocks when hitting them on the side.
![V0 of frame prediction](output.gif)

Next step is simulating the game "Breakout" in python. We need to first create a human playable version of the game and then allow a reinforcement learning agent to take the wheel.

![Breakout human playable](human_playable_breakout.gif)


# How to use
This is work in progress and still quite raw. The functionality can be explored via the jupyter notebooks.
