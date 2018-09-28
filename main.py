import argparse, torch
import numpy as np
import core.utils as utils
import cpommerman, os
import pommerman
from pommerman.agents import BaseAgent, SimpleAgent, RandomAgent
from pommerman import constants
import core.models as models
from core.models import Conv_model, LayerNorm
import core.expert_iteration as eXit
from core.agent import Agent

NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)
NUM_CHANNELS = 18
os.environ["CUDA_VISIBLE_DEVICES"]='3'
LOAD_TRAINED_MODEL = True


class Evaluate:
    def __init__(self, agent):
        # Test pool and Env
        self.pool = [agent, SimpleAgent(), SimpleAgent(), SimpleAgent(), ]
        #self.pool = [agent, RandomAgent(), RandomAgent(), RandomAgent(), ]
        self.env = pommerman.make('PommeFFACompetition-v0', self.pool)

    def featurize(self, obs):
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

    def evaluate_policy(self, model, num_evals):
        all_rewards = []; all_lengths = []
        for _ in range(num_evals):
            obs = self.env.reset(); length = 0; done = False
            while not done:
                # Take action
                probs, values = model.predict(np.expand_dims(self.featurize(obs[0]), axis=0))
                action = np.random.choice(NUM_ACTIONS, p=probs.flatten())

                # make other agents act
                actions = self.env.act(obs)
                # add my action to list of actions
                actions[0] = action
                # step environment
                obs, rewards, done, info = self.env.step(actions)
                length += 1
            all_rewards.append(rewards[0]); all_lengths.append(length)

        return np.mean(all_lengths), np.mean(all_rewards)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='conv3x256value.h5')
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_false", default=False)
    # runner params
    parser.add_argument('--num_epochs', type=int, default=10000000)
    parser.add_argument('--num_runners', type=int, default=1)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=50)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.4)
    args = parser.parse_args()

    #Trackers
    expert_tracker = utils.Tracker('log', ['expert_reward'], '.csv')  # Initiate tracker
    net_tracker = utils.Tracker('log', ['net_reward'], '.csv')  # Initiate tracker

    #Load/Create Model
    if LOAD_TRAINED_MODEL: model = torch.load('pretrained_model.pth')
    else: model = models.Conv_model(z_dim=250)
    model.cuda()
    #Initialize learning agent
    agent = Agent()
    #Initialize expert
    expert = eXit.Expert(0, args)
    #Initialize imitation engine and evaluator
    imitation_engine = models.Imitation(model)
    evaluator = Evaluate(agent)

    #TRAINING LOOP
    for epoch in range(1, args.num_epochs):
        # Get data from expert rollout
        expert_length, expert_reward, x_feats, y_probs, y_values, elapsed = expert.train(model)

        #Train model using the data
        model = imitation_engine.imitate(model, x_feats, y_probs, y_values)

        #Test on test set
        test_length, test_reward = evaluator.evaluate_policy(model, num_evals=5)

        #Update to tracker
        expert_tracker.update([expert_reward], epoch)
        net_tracker.update([test_reward], epoch)
        print ('Epoch', epoch, 'EXP_REW ' '%.2f'%expert_tracker.all_tracker[0][1], ' NET_REW ', '%.2f'%net_tracker.all_tracker[0][1], ' Exp_len', '%.2f'%expert_length, 'Net_len', '%.2f'%test_length)

        #Save periodically
        if epoch % 25 == 0:
            print ('SAVING MODEL')
            torch.save(model, 'net.pth')












