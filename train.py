import argparse
from trainer.MA import MADDPG
import numpy as np
from env.environment import MultiAgentEnv
from env import simple_tag
import torch

def make_env(scenario_name, args):
    #scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario = simple_tag.Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_space, action_space, args):
    trainers = []
    for i in range(num_adversaries):
        trainers.append(MADDPG(i, obs_space, action_space, args))
    for i in range(num_adversaries, env.n):
        trainers.append(MADDPG(i, obs_space, action_space, args))
    return trainers


def evaluate(args, env, trainers):
    out = []
    for e in range(args.evaluate_episodes):
        obs_n = env.reset()
        rew = 0
        for t in range(args.evaluate_steps):
            action_n = [agent.act(torch.tensor(o)) for agent, o in zip(trainers, obs_n)]
            next_obs_n, rew_n, done_n, info_n = env.step(action_n)
            rew += rew_n[0]
            obs_n = next_obs_n
        out.append(rew)
        print(f'Evaluation rewards : {rew}')
    return sum(out)/args.evaluate_episodes



def train(args):
    env = make_env('simple_tag', args)
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = np.zeros((env.n, 1)) # (num of agents, 1)
    obs_space = env.observation_space
    action_space = env.action_space
    num_adv = args.num_adversaries
    num_episodes = args.num_episodes
    batch_size = args.batch_size
    train_steps_per_episode = args.train_steps_per_episode
    # get initial obs
    obs_n = env.reset()
    # setup trainer
    trainers = get_trainers(env, num_adv, obs_space, action_space, args)

    # train
    num_steps_collected = 0
    for n in range(num_episodes):
        print(f'Episode {n} begins')

        # num_agent x size 
        obs = [[] for _ in range(env.n)]
        actions = [[] for _ in range(env.n)]
        rews = [[] for _ in range(env.n)]
        next_obs = [[] for _ in range(env.n)]
        done = [[] for _ in range(env.n)]

        for t in range(train_steps_per_episode):
            action_n = [agent.act(torch.tensor(o)) for agent, o in zip(trainers, obs_n)]
            next_obs_n, rew_n, done_n, info_n = env.step(action_n)
            done = all(done_n)

            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], next_obs_n[i])
                #obs[i].append(obs_n[i])
                #actions[i].append(action_n[i])
                #rews[i].append(rew_n[i])
                #next_obs[i].append(next_obs_n[i])
            if done:
                break
            obs_n = next_obs_n
        
        # save replay buffer
        #for i, agent in enumerate(trainers):
        #    agent.experience(obs[i], actions[i], rews[i], next_obs[i])

        for i, rew in enumerate(rews):
            episode_rewards[-1] += np.sum(rew)
            agent_rewards[i][-1] += np.sum(rew)
                    
        # update
        for i, agent in enumerate(trainers):
            current_agent = i
            agent.update(trainers, batch_size)


        # save model, display training output
        if n % 100 == 0:
            ret = evaluate(args, env, trainers)
            print(f'Average returns is {ret}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_type", type=str, default="mpe", help="name of scenario")
    parser.add_argument("--scenario", type=str, default="simple_v2", help="name of the scenario script")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--batch_size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--train_steps_per_episode", type=int, default=100)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--max_path_length", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--evaluate_steps", type=int, default=1000)
    parser.add_argument("--evaluate_episodes", type=int, default=1000)
    
    args = parser.parse_args()
    train(args)