
import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *

np.set_printoptions(precision=3)


def policy_iteration(env, gamma, max_iteration, tol):

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)

    flag = False
    count = 0
    while not flag:
        V = policy_evaluation(env, policy, gamma, max_iteration, tol)
        policy, flag = policy_improvement(env, V, policy, gamma)
        count = count + 1
    return V, policy


def policy_evaluation(env, policy, gamma, max_iteration, tol):

    V = np.zeros(env.nS)
    t = 0
    while True:
        delta = 0
        t = t + 1
        for i in range(env.nS):
            temp = V[i]
            V[i] = env.P[i][policy[i]][0][2] + gamma * V[env.P[i][policy[i]][0][1]]
            delta = max(delta, abs(V[i]-temp))
        if delta < tol:
            break
        if t > max_iteration:
            break
    return V


def policy_improvement(env, value_from_policy, policy, gamma):

    policy_stable = True
    for i in range(env.nS):
        temp = policy[i]
        pt = 0
        for j in range(env.nA):
            if env.P[i][j][0][2] + gamma * value_from_policy[env.P[i][j][0][1]] > pt:
                policy[i] = j
                pt = env.P[i][j][0][2] + gamma * value_from_policy[env.P[i][j][0][1]]
        if temp != policy[i]:
            policy_stable = False


    return policy, policy_stable



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
        print(ob,rew)
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
            ob, reward, done, info = env.step(a)
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
        print(s)
        action = int(input("Please type in the next action:"))
        s, r, done, info = env.step(action)

        print(s)
        print(r)
        print(done)
        print(info)

    env.close()


if __name__ == '__main__':

    env = gym.make("Assignment1-Taxi-v2")
    print(env.__doc__)
    V_pi, policy_pi = policy_iteration(env, gamma=0.95, max_iteration=6000, tol=1e-5)

    scores = avg_performance(env, policy_pi)
    print(V_pi)
    print(policy_pi)
    print(scores)

 