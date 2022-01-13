import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":



    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0

        #TODO PERFORM Q LEARNING
        state, done = env.reset(), False

        while not done:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(np.array([Q_table[(state,i)] for i in range(env.action_space.n)]))

            new_state, reward, done, info = env.step(action)

            if done:
                q_done = reward - Q_table[state, action]
                Q_table[state, action] += LEARNING_RATE * q_done
            else: 
                pi = np.max(np.array([Q_table[(new_state, i)] for i in range(env.action_space.n)]))
                q_next = reward + DISCOUNT_FACTOR * pi - Q_table[state, action]
                Q_table[state, action] += LEARNING_RATE * q_next

            state = new_state
            
        EPSILON *= EPSILON_DECAY
        episode_reward_record.append(reward)

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################







