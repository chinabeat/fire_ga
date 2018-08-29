import tensorflow as tf
import numpy as np
import random
from collections import deque
import fire_ga_version_7

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = len(env.state)
        # self.action_dim = env.action_space.n
        self.action_dim = env.action_dim

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 256])
        b1 = self.bias_variable([256])
        W2 = self.weight_variable([256, 256])
        b2 = self.bias_variable([256])
        W3 = self.weight_variable([256, 512])
        b3 = self.bias_variable([512])
        W4 = self.weight_variable([512, self.action_dim])
        b4 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer_1 = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        h_layer_2 = tf.nn.relu(tf.matmul(h_layer_1, W2) + b2)
        h_layer_3 = tf.nn.relu(tf.matmul(h_layer_2, W3) + b3)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer_3, W4) + b4
        # self.Q_value = tf.nn.softmax(tf.matmul(h_layer_3,W4) + b4)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def save(self):
        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.session, './model_version_7/model_ckpt')
        print("The saved path of checkpoint is {}.".format(self.save_path))

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

            # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])


# ---------------------------------------------------------
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = fire_ga_version_7.env()
    agent = DQN(env)
    # env_initial = np.ones(10)
    # agent = DQN()
    success_count = 0
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        print("state :{}".format(state))
        print("episode: {}".format(episode))
        # Train
        for step in range(STEP):
            state = state
            # print("state :{}".format(state))
            action = agent.egreedy_action(state)  # e-greedy action for train
            # next_state,reward,done,_ = env.step(action)
            # print("state :{}".format(state))
            next_state, reward, done = env.step(action)
            # print("state :{}".format(state))
            # Define reward for agent
            # reward_agent = -1 if done else 0.1
            print("state:{},action:{},reward:{},next_state:{},done:{}".format(state, action, reward, next_state, done))
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            print("########## start test ##########")
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    print("state :{}".format(state))
                    action = agent.action(state)  # direct action for test
                    # print("state: {}, action: {}".format(state,action))
                    # state,reward,done,_ = env.step(action)
                    state, reward, done = env.step(action)
                    print("state: {}, reward: {}, done: {}".format(state, reward, done))
                    total_reward += reward
                    print("total_reward: {}".format(total_reward))
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            print("############# end test ##############")
            if ave_reward >= 900:
                success_count += 1
            else:
                success_count = 0
            print("episode: {}, ave_reward: {}, success_count: {}".format(episode, ave_reward, success_count))
            if success_count == 5:
                print("SUCCESS!!!")
                agent.save()
                break


if __name__ == '__main__':
    main()
