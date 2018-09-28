
import numpy as np
import time
import cpommerman, os
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent, RandomAgent
from pommerman import constants


NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18
os.environ["CUDA_VISIBLE_DEVICES"]='3'
LOAD_TRAINED_MODEL = False


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

    # assert len(maps) == NUM_CHANNELS
    return np.stack(maps, axis=2)

def argmax_tiebreaking_axis1(Q):
    # find the best action with random tie-breaking
    mask = (Q == np.max(Q, axis=1, keepdims=True))
    return np.array([np.random.choice(np.flatnonzero(m)) for m in mask], dtype=np.uint8)


def argmax_tiebreaking(Q):
    # find the best action with random tie-breaking
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)


class MCTSNode(object):
    def __init__(self, p):
        # values for 6 actions
        self.Q = np.zeros((NUM_AGENTS, NUM_ACTIONS))
        self.W = np.zeros((NUM_AGENTS, NUM_ACTIONS))
        self.N = np.zeros((NUM_AGENTS, NUM_ACTIONS), dtype=np.uint32)
        assert p.shape == (NUM_AGENTS, NUM_ACTIONS)
        self.P = p

    def actions(self, args):
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

    def search(self, root, num_iters, temperature, args):
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
                    actions = node.actions(args)
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

    def rollout(self, env, args):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        x_feats = [];
        y_probs = [];
        temp_values = []

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
            pi = self.search(root, args.mcts_iters, args.temperature, args)
            pi = pi[self.agent_id]
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            x_feats.append(featurize(obs[self.agent_id]));
            y_probs.append(pi)

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
            # print("Agent:", self.agent_id, "Step:", length, "Actions:", [constants.Action(a).name for a in actions], "Probs:", [round(p, 2) for p in pi], "Rewards:", rewards, "Done:", done)

        reward = rewards[self.agent_id]
        # Also compute values using discounting
        y_values = []
        for entry in temp_values:
            y_values.append(entry[0] * (length + 1 - entry[1]) * 0.9)

        return length, reward, rewards, x_feats, y_probs, y_values

    def act(self, obs, action_space):
        # TODO
        assert False


class Expert:
    def __init__(self, agent_id, args):
        self.args = args
        #Initialize the expert and the pool
        self.expert = MCTSAgent(None, agent_id=agent_id)
        self.pool = [SimpleAgent(), SimpleAgent(), SimpleAgent(), ]
        #self.pool = [RandomAgent(), RandomAgent(), RandomAgent(), ]
        self.pool.insert(agent_id, self.expert)
        self.env = pommerman.make('PommeFFACompetition-v0', self.pool)

    def argmax_tiebreaking_axis1(self, Q):
        # find the best action with random tie-breaking
        mask = (Q == np.max(Q, axis=1, keepdims=True))
        return np.array([np.random.choice(np.flatnonzero(m)) for m in mask], dtype=np.uint8)

    def argmax_tiebreaking(self, Q):
        # find the best action with random tie-breaking
        idx = np.flatnonzero(Q == np.max(Q))
        assert len(idx) > 0, str(Q)
        return np.random.choice(idx)

    def train(self, model):
        self.expert.model = model
        start_time = time.time()
        length, reward, rewards, x_feats, y_probs, y_values = self.expert.rollout(self.env, self.args)
        elapsed = time.time() - start_time

        return length, reward, x_feats, y_probs, y_values, elapsed







