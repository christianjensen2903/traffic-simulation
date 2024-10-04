import numpy as np
import sumo_env


NORTH = 0
EAST_ = 1
SOUTH = 2
WEST_ = 3

ALL = 0
STRAIGHT = 1
LEFT = 2
RIGHT = 3
STRAIGHT_RIGHT = 4
STRAIGHT_LEFT = 5


if __name__ == "__main__":
    env = sumo_env.SumoEnv(intersection_path="intersections")
    env.visualize = True
    env.reset()


    action = np.zeros((4,6))
    action[EAST_, STRAIGHT_LEFT] = 1

    loss = 0
    while True:
        input()
        obs, reward, done, _, _ = env.step(action.flatten())
        loss += reward
        if done:
            break

    print("Total reward:", loss)

    env.close()
