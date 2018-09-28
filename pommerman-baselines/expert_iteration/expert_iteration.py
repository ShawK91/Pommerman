import argparse
import multiprocessing
import numpy as np
import time

import cpommerman, os
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent, RandomAgent
from pommerman import constants

import tensorflow as tf
import keras.backend as K
from keras.models import load_model

import models as models

NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18
os.environ["CUDA_VISIBLE_DEVICES"]='3'
LOAD_TRAINED_MODEL = False





class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros((NUM_AGENTS, NUM_ACTIONS))
        self.W = np.zeros((NUM_AGENTS, NUM_ACTIONS))
        self.N = np.zeros((NUM_AGENTS, NUM_ACTIONS), dtype=np.uint32)
        assert p.shape == (NUM_AGENTS, NUM_ACTIONS)
        self.P = p

    def actions(self):
        U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N, axis=1, keepdims=True)) / (1 + self.N)
        return argmax_tiebreaking_axis1(self.Q + U)

    def update(self, actions, rewards):
        assert len(actions) == len(rewards)
        self.W[range(NUM_AGENTS), actions] += rewards
        self.N[range(NUM_AGENTS), actions] += 1
        self.Q[range(NUM_AGENTS), actions] = self.W[range(NUM_AGENTS), actions] / self.N[range(NUM_AGENTS), actions]

    def probs(self, temperature=1):
        if temperature == 0:
            p = np.zeros(self.N.shape)
            idx = argmax_tiebreaking_axis1(self.N)
            p[range(NUM_AGENTS), idx] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt, axis=1, keepdims=True)


class MCTSAgent(BaseAgent):
    def __init__(self, model, agent_id=0):
        super().__init__()
        self.model = model
        self.agent_id = agent_id
        self.env = cpommerman.make()
        self.reset_tree()

    def reset_tree(self):
        self.tree = {}

    def search(self, root, num_iters, temperature=1):
        # remember current game state
        self.env.set_json_info(root)
        root = self.env.get_state()

        for i in range(num_iters):
            # restore game state to root node
            self.env.set_state(root)
            # serialize game state
            state = root

            trace = []
            done = False
            # fetch rewards so we know which agents are alive
            rewards = self.env.get_rewards()
            while not done:
                if state in self.tree:
                    node = self.tree[state]
                    # choose actions based on Q + U
                    actions = node.actions()
                    # use Stop action for all dead agents to reduce tree size
                    actions[rewards != 0] = constants.Action.Stop.value
                else:
                    # initialize action probabilities with policy network
                    feats = self.env.get_features()
                    probs, values = self.model.predict(feats)

                    # add new node to the tree
                    self.tree[state] = MCTSNode(probs)

                    # for alive agents use state value, for others use returned reward
                    rewards[rewards == 0] = values[rewards == 0, 0]

                    # stop at leaf node
                    break

                # step environment forward
                self.env.step(actions)
                rewards = self.env.get_rewards()
                done = self.env.get_done()
                trace.append((node, actions, rewards))

                # fetch next state
                state = self.env.get_state()

            # update tree nodes with rollout results
            for node, actions, rews in reversed(trace):
                # use the reward of the last timestep where it was non-null
                rewards[rews != 0] = rews[rews != 0]
                node.update(actions, rewards)
                rewards *= args.discount

        # return action probabilities
        return self.tree[root].probs(temperature)

    def rollout(self, env):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        x_feats = []; y_probs = []; temp_values = []

        # guarantees that we are not called recursively
        # and episode ends when this agent dies
        env.training_agent = self.agent_id
        obs = env.reset()

        length = 0
        done = False
        while not done:
            if args.render:
                env.render()

            root = env.get_json_info()
            # do Monte-Carlo tree search
            pi = self.search(root, args.mcts_iters, args.temperature)
            pi = pi[self.agent_id]
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            x_feats.append(featurize(obs[self.agent_id])); y_probs.append(pi)

            # ensure we are not called recursively
            assert env.training_agent == self.agent_id
            # make other agents act
            actions = env.act(obs)
            # add my action to list of actions
            actions.insert(self.agent_id, action)
            # step environment
            obs, rewards, done, info = env.step(actions)
            assert self == env._agents[self.agent_id]
            length += 1
            temp_values.append([rewards[self.agent_id], length])
            #print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        #Also compute values using discounting
        y_values = []
        for entry in temp_values:
            y_values.append(entry[0] * (length + 1 - entry[0]) * 0.9 )

        return length, reward, rewards, x_feats, y_probs, y_values

    def act(self, obs, action_space):
        # TODO
        assert False

def profile_runner(id, num_episodes, fifo, _args):
    import cProfile
    command = """runner(id, num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)



def init_tensorflow():
    # make sure TF does not allocate all memory
    # NB! this needs to be done also in subprocesses!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

def featurize(obs):
    # TODO: history of n moves?
    board = obs['board']

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e.value for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    #assert len(maps) == NUM_CHANNELS
    return np.stack(maps, axis=2)

def train(num_episodes, _args, model):
    all_rewards = []
    all_lengths = []
    all_elapsed = []

    # make args accessible to MCTSAgent
    global args
    args = _args

    # make sure TF does not allocate all memory
    init_tensorflow()

    # make sure agents play at all positions
    agent_id = 0
    expert = MCTSAgent(model, agent_id=agent_id)

    # create environment with three SimpleAgents
    #agents = [SimpleAgent(), SimpleAgent(),SimpleAgent(),]
    expert_pool = [SimpleAgent(),RandomAgent(),RandomAgent(),]
    agent_pool =[SimpleAgent(),RandomAgent(),RandomAgent(),]# expert_pool[:]
    expert_pool.insert(agent_id, expert)
    expert_env = pommerman.make('PommeFFACompetition-v0', expert_pool)

    train_agent = Agent(model)
    agent_pool.insert(agent_id, train_agent)
    agent_env = pommerman.make('PommeFFACompetition-v0', agent_pool)

    #Train using expert_pool in expert environment
    for i in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward, rewards, x_feats, y_probs, y_values = expert.rollout(expert_env)
        elapsed = time.time() - start_time
        # add data samples to log
        #fifo.put((length, reward, rewards, agent_id, elapsed))

        #Train the agent net to compile expert decision
        #callbacks = [ModelCheckpoint('conv256_single_value10_disc0.9_best.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'), EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')]
        model.fit(np.array(x_feats), [np.array(y_probs), np.array(y_values)], batch_size=12, epochs=1)#, validation_data=(x_test, [p_test, v_test]))

        length, reward = evaluate_policy(model, agent_env, num_evals=10)

        # do logging in the main process
        all_rewards.append(reward)
        all_lengths.append(length)
        all_elapsed.append(elapsed)
        print("Episode:", i, "Average Reward:", '%.2f' %np.mean(all_rewards[-10:]), "Length:", '%.2f' %np.mean(all_lengths[-10:]), "Time per step:", '%.3f' % (np.sum(all_elapsed[-10:]) / np.sum(all_lengths[-10:])))




def evaluate_policy(model, env, num_evals):
    all_rewards = []; all_lengths = []
    for _ in range(num_evals):
        obs = env.reset()

        length = 0; done = False
        while not done:
            #Take action
            probs, values = model.predict(np.expand_dims(featurize(obs[0]), axis=0))
            action = np.random.choice(NUM_ACTIONS, p=probs.flatten())

            # make other agents act
            actions = env.act(obs)
            # add my action to list of actions
            actions[0] = action
            # step environment
            obs, rewards, done, info = env.step(actions)
            length += 1
        all_rewards.append(rewards[0]); all_lengths.append(length)

    return np.mean(all_lengths), np.mean(all_rewards)


class Agent(BaseAgent):
    def __init__(self, model, agent_id=0):
        super().__init__()
        self.model = model
        self.agent_id = agent_id


    def act(self, obs, action_space):
        # TODO
        pass






# class Expert_Iteration:
#     def __init__(self):
#         pass
#
#     def train(self):
#
#         # make sure TF does not allocate all memory
#         init_tensorflow()
#
#         # make sure agents play at all positions
#         agent_id = id % NUM_AGENTS
#         expert = MCTSAgent(args.model_file, agent_id=agent_id)
#
#         # create environment with three SimpleAgents
#         agents = [SimpleAgent(),SimpleAgent(),SimpleAgent(),]
#         agents = [RandomAgent(),RandomAgent(),RandomAgent(),]
#         agent_id = id % NUM_AGENTS
#         agents.insert(agent_id, agent)
#
#         env = pommerman.make('PommeFFACompetition-v0', agents)
#
#         for i in range(num_episodes):
#             # do rollout
#             start_time = time.time()
#             length, reward, rewards = agent.rollout(env)
#             elapsed = time.time() - start_time
#             # add data samples to log
#             fifo.put((length, reward, rewards, agent_id, elapsed))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='conv3x256value.h5')
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_false", default=False)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=10000000)
    parser.add_argument('--num_runners', type=int, default=1)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=50)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.4)
    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"
    if LOAD_TRAINED_MODEL: model = load_model(args.model_file)
    else: model = models.init_model()


    # # use spawn method for starting subprocesses
    # ctx = multiprocessing.get_context('spawn')
    #
    # # create fifos and processes for all runners
    # fifo = ctx.Queue()
    # for i in range(args.num_runners):
    #     process = ctx.Process(target=profile_runner if args.profile else train, args=(i, args.num_episodes // args.num_runners, fifo, args, model))
    #     process.start()

    train(args.num_episodes, args, model)

    # # do logging in the main process
    # all_rewards = []
    # all_lengths = []
    # all_elapsed = []
    # for i in range(args.num_episodes):
    #     # wait for a new trajectory
    #     length, reward, rewards, agent_id, elapsed = fifo.get()
    #
    #
    #     all_rewards.append(reward)
    #     all_lengths.append(length)
    #     all_elapsed.append(elapsed)
    #     print("Episode:", i, "Average Reward:", '%.2f' %np.mean(all_rewards[-10:]), "Length:", '%.2f' %np.mean(all_lengths[-10:]), "Time per step:", '%.3f' % (np.sum(all_elapsed[-10:]) / np.sum(all_lengths[-10:])))


