
import numpy as np
import gym
import time
import random
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
from taxi_envs import *


def QLearning(env, num_episodes, gamma, lr, e):

    Q = np.zeros((env.nS, env.nA))
    cumureward = np.zeros(num_episodes)
    length = np.zeros(num_episodes)
    progress = np.zeros(num_episodes)
    count = 0
    while True:
        if count >= num_episodes:
            break
        cumureward[count] = 0
        length[count] = 0
        s = random.randint(0, env.nS - 1)
        a = random.randint(0, env.nA - 1)
        while True:
            length[count] = length[count] + 1
            agreedy = np.argmax(Q[s])
            if random.random() < 1 - e:
                a = agreedy
            else:
                a = random.randint(0, env.nA - 1)
            s_next = env.P[s][a][0][1]
            r = env.P[s][a][0][2]
            cumureward[count] = cumureward[count] + gamma * r
            Q[s][a] = Q[s][a] + lr * (r + gamma * max(Q[s_next]) - Q[s][a])
            s = s_next
            if env.P[s][a][0][3]:
                break
        if count != 0:
            progress[count] = (progress[count - 1] * count + cumureward[count]) / (count + 1)
        else:
            progress[count] = cumureward[count]
        count = count + 1


    # plt.plot(cumureward)
    # plt.xlabel('Episode')
    # plt.ylabel('Cumu. Reward of Episode')
    # plt.show()

    # plt.plot(progress)
    # plt.xlabel('Episode')
    # plt.ylabel('Avg. Reward of Episode')
    # plt.show()

    # plt.plot(length)
    # plt.xlabel('Episode')
    # plt.ylabel('Length of Episode')
    # plt.show()

    return Q


def SARSA(env, num_episodes, gamma, lr, e):

    Q = np.zeros((env.nS, env.nA))
    cumureward = np.zeros(num_episodes)
    length = np.zeros(num_episodes)
    progress = np.zeros(num_episodes)
    count = 0
    while True:
        if count >= num_episodes:
            break
        #print(count)
        cumureward[count] = 0
        length[count] = 0
        s = random.randint(0, env.nS - 1)
        a = random.randint(0, env.nA - 1)

        while True:
            length[count] = length[count] + 1
            s_next = env.P[s][a][0][1]
            r = env.P[s][a][0][2]
            cumureward[count]  = cumureward[count] + gamma * r
            agreedy = np.argmax(Q[s_next])
            if random.random() < 1 - e:
                a_next = agreedy
            else:
                a_next = random.randint(0, env.nA - 1)
            Q[s][a] = Q[s][a] + lr * (r + gamma * Q[s_next][a_next] - Q[s][a])
            s = s_next
            a = a_next
            if env.P[s][a][0][3]:
                break
        if count != 0:
            progress[count] = (progress[count - 1] * count + cumureward[count]) / (count + 1)
        else:
            progress[count] = cumureward[count]
        count = count + 1

    # plt.plot(cumureward)
    # plt.xlabel('Episode')
    # plt.ylabel('Cumu. Reward of Episode')
    # plt.show()

    # plt.plot(progress)
    # plt.xlabel('Episode')
    # plt.ylabel('Avg. Reward of Episode')
    # plt.show()

    # plt.plot(length)
    # plt.xlabel('Episode')
    # plt.ylabel('Length of Episode')
    # plt.show()

    return Q


def render_episode_Q(env, Q):

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print ("Episode reward: %f" %episode_reward)



def main():
    env = gym.make("Assignment1-Taxi-v2")
    Q_QL = QLearning(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    Q_Sarsa = SARSA(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    print(Q_QL)
    print(Q_Sarsa)


if __name__ == '__main__':
    main()
