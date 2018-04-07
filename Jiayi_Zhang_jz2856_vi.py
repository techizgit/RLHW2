

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *

np.set_printoptions(precision=3)


def value_iteration(env, gamma, max_iteration, tol):


    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)
    t = 0
    while True:
        delta = 0
        t = t + 1
        for i in range(env.nS):
            temp = V[i]
            for j in range(env.nA):
                tempv = env.P[i][j][0][2] + gamma * V[env.P[i][j][0][1]]
                if tempv > V[i]:
                    V[i] = tempv
            delta = max(delta, abs(temp - V[i]))
        if delta < tol:
            break
        if t > max_iteration:
            break
    policy  = extract_policy(env, V, gamma)
    return V, policy


def extract_policy(env, v, gamma):

    policy = np.zeros(env.nS, dtype=int)
    for i in range(env.nS):
        pt = -99999
        for j in range(env.nA):
            if env.P[i][j][0][2] + gamma * v[env.P[i][j][0][1]] > pt:
                policy[i] = j
                pt = env.P[i][j][0][2] + gamma * v[env.P[i][j][0][1]]
    return policy


def example(env):

    env.seed(0)
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render()


def render_episode(env, policy):

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render()
    print("Episode reward: %f" % episode_reward)


def avg_performance(env, policy):

    sum_reward = 0.
    episode = 100
    max_iteration = 6000
    for i in range(episode):
        done = False
        ob = env.reset()

        for j in range(max_iteration):
            a = policy[ob]
            ob, reward, done, _ = env.step(a)
            sum_reward += reward
            if done:
                break

    return sum_reward / i

def main():

    GAME = "Assignment1-Taxi-v2"
    env = gym.make(GAME)
    n_state = env.observation_space.n
    n_action = env.action_space.n
    env = Monitor(env, "taxi_simple", force=True)

    s = env.reset()
    steps = 100
    for step in range(steps):
        env.render()
        action = int(input("Please type in the next action:"))
        s, r, done, info = env.step(action)
        print(s)
        print(r)
        print(done)
        print(info)

    env.close()


if __name__ == "__main__":

    env = gym.make("Assignment1-Taxi-v2")
    print(env.__doc__)
    V_vi, policy_vi = value_iteration(env, gamma=0.95, max_iteration=6000, tol=1e-5)
    scores = avg_performance(env, policy_vi)
    print(V_vi)
    print(policy_vi)
    print(scores)

