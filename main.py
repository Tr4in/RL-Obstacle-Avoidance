'''
Implementation inspired by:

1. https://www.youtube.com/watch?v=H9uCYnG3LlE, last visited: 26.01.2022
2. https://github.com/xie9187/Monocular-Obstacle-Avoidance, last visited: 26.01.2022

'''

from collections import deque
from numpy.core.fromnumeric import size
import torch
import torch.optim as optim
from environment import Environment
from agent import Agent, DeepQNetwork
import win32pipe, win32file
import numpy as np
import time
from matplotlib import pyplot as plt
from prioritized_experience_replay_buffer import PrioritizedExperienceReplayBuffer
from experience_replay_buffer import ExperienceReplayBuffer
from ue_communicator import UECommunicator

MODE = 0
TRAINED_MODEL_PATH = './trained_q_model/q_model_episode_50.pth'
PLOT_REWARDS_PATH = './plots/plot_episode_{}'
NUM_EPISODES = 1000
NUM_TEST_EPISODES = 2000
NUM_TIMESTEPS = 1000
NUM_ACTIONS = 6
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 8
CONSECUTIVE_IMAGES = 4
OSERVATION_EPISODES = 10
UPDATE_TARGET = 50
SAVE_MODELS = 50
EPSILON_DEC_TIMESTEPS = 30000
NUM_LASER = 9
LAYER_NORM = False
PER = False

REWARDS = []
AVERAGE_REWARDS = []
GAMMA = 0.995
LEARNING_RATE = 0.0001
EPSILON = 0.7
EPSILON_END = 0.001
EPSILON_DEC = (EPSILON - EPSILON_END) / EPSILON_DEC_TIMESTEPS
IMAGE_SHAPE = (192, 320)
AVERAGE_NUM = 10
ALPHA = 0.7
BETA = 0.5
BETA_INCREMENT = (1 - BETA) / EPSILON_DEC_TIMESTEPS 


def start_training(start_episode = None):
    start_episode = 0 if start_episode is None else start_episode
    ue_communicator.wait_for_environment_loading()

    for episode in range(start_episode + 1, NUM_EPISODES + 1):
        print('Episode {}'.format(episode))

        done = False
        observation = environment.reset()
        total_reward = 0
        step = 0

        while not done and step < NUM_TIMESTEPS:
            start = time.time()

            if episode <= OSERVATION_EPISODES:
                action = np.random.choice(NUM_ACTIONS)
            else:
                action = agent.get_next_action(observation, True)

            reward, next_observation, done = environment.step(action)

            REPLAY_BUFFER.add_experience((observation, next_observation, action, reward, done))

            observation = next_observation
            total_reward += reward
            step = step + 1

            if episode > OSERVATION_EPISODES: 
                agent.optimize_q_network(REPLAY_BUFFER, episode)

            end = time.time()

            diff = end - start
            print('Step-Time: {}'.format(diff))
        
        REWARDS.append([episode, total_reward])
        average_reward = (sum([reward for _, reward in REWARDS[-AVERAGE_NUM:]]) / AVERAGE_NUM) if len(REWARDS) >= AVERAGE_NUM else (sum([reward for _, reward in REWARDS]) / max(1, len(REWARDS)))

        AVERAGE_REWARDS.append([episode, average_reward])
        show_and_save_plot(episode)

        if episode % SAVE_MODELS == 0:
            save_trained_model(episode)
            
        if episode % UPDATE_TARGET == 0:
            agent.update_target()
    
    save_trained_model(NUM_EPISODES)
    show_plot()


def test_model():
    for episode in range(1, NUM_TEST_EPISODES + 1):
        print('Test Episode {}'.format(episode))

        done = False
        observation = environment.reset()
        total_reward = 0
        step = 0

        while not done and step < NUM_TIMESTEPS:
            action = agent.get_next_action(observation, False)
            reward, next_observation, done = environment.step(action)
            observation = next_observation
            total_reward += reward
            step = step + 1
        
        AVERAGE_REWARDS.append([episode, total_reward])
     

def load_models_and_last_episode(model_path):
    checkpoint = torch.load(model_path)
    
    main_model = DeepQNetwork(LEARNING_RATE, NUM_ACTIONS)
    main_model.load_state_dict(checkpoint['main_state_dict'])
    main_loss = checkpoint['main_loss']
    main_optimizer = optim.Adam(main_model.parameters(), lr=LEARNING_RATE)
    main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
    main_model.set_optimizer(main_optimizer)
    main_model.set_loss(main_loss)

    target_model = DeepQNetwork(LEARNING_RATE, NUM_ACTIONS)
    target_model.load_state_dict(checkpoint['target_state_dict'])
    target_loss = checkpoint['target_loss']
    target_optimizer = optim.Adam(target_model.parameters(), lr=LEARNING_RATE)
    target_optimizer.load_state_dict(checkpoint['target_optimizer_state_dict'])
    target_model.set_optimizer(target_optimizer)
    target_model.set_loss(target_loss)

    episode = checkpoint['epoch']

    return main_model, target_model, episode

def save_trained_model(episode):
    agent.save_model(episode)

def show_plot():
    plt.ion()
    plt.show()
    plt.figure(num='Episodes and average rewards of the previous 10 episodes')
    plt.title('Episodes and Average-Rewards') 
    plt.xlabel('Episodes') 
    plt.ylabel('Average-Reward Previous 10 Episodes')

    plt.plot(*zip(*AVERAGE_REWARDS), c='black')
    plt.draw()
    plt.pause(0.001)

def show_and_save_plot(episode):
    show_plot()
    plt.savefig(PLOT_REWARDS_PATH.format(episode))
    #plt.clf()
'''
def save_plot(episode):
    plt.figure(num='Episodes and average rewards of the previous 10 episodes')
    plt.title('Episodes and Average-Reward') 
    plt.xlabel('Episodes') 
    plt.ylabel('Average-Reward Previous 10 Episodes') 
    plt.plot(*zip(*AVERAGE_REWARDS), c='black')
    plt.savefig(PLOT_REWARDS_PATH.format(episode))
'''

pipe = win32pipe.CreateNamedPipe(
    r'\\.\pipe\ABC',
    win32pipe.PIPE_ACCESS_DUPLEX,
    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
    win32pipe.PIPE_UNLIMITED_INSTANCES, 65536, 65536,
    5000,
    None)
try:
    print('Waiting For Unreal Engine Agent')
    win32pipe.ConnectNamedPipe(pipe, None)
    print('Got Agent')

    ue_communicator = UECommunicator(pipe)
    environment = Environment(ue_communicator, CONSECUTIVE_IMAGES, NUM_TIMESTEPS, NUM_LASER)

    if PER:
        REPLAY_BUFFER = PrioritizedExperienceReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, ALPHA, BETA, BETA_INCREMENT)
    else:
        REPLAY_BUFFER = ExperienceReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE)

    if MODE == 0:
        agent = Agent(GAMMA, EPSILON, LEARNING_RATE, IMAGE_SHAPE, REPLAY_BUFFER_SIZE, BATCH_SIZE, NUM_ACTIONS, EPSILON_END, EPSILON_DEC, layer_norm = LAYER_NORM, per = PER)
        start_training()
    elif MODE == 1:
        main_network, target_network, episode = load_models_and_last_episode(TRAINED_MODEL_PATH)
        agent = Agent(GAMMA, EPSILON - ((episode - OSERVATION_EPISODES) * EPSILON_DEC), LEARNING_RATE, IMAGE_SHAPE, REPLAY_BUFFER_SIZE, BATCH_SIZE, NUM_ACTIONS, EPSILON_END, EPSILON_DEC, target_network, main_network, LAYER_NORM, PER)
        start_training(episode)
    elif MODE == 2:
        main_network, target_network, _ = load_models_and_last_episode(TRAINED_MODEL_PATH)
        agent = Agent(GAMMA, EPSILON, LEARNING_RATE, IMAGE_SHAPE, REPLAY_BUFFER_SIZE, BATCH_SIZE, NUM_ACTIONS, EPSILON_END, EPSILON_DEC, main_network, target_network, LAYER_NORM, PER)
        test_model()

    print('finished now')
finally:
    win32file.CloseHandle(pipe)
