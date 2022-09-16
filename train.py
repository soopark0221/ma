import argparse
from trainer.MA import MADDPG
import numpy as np
from env.environment import MultiAgentEnv
from env import simple_tag
import torch
import os
import imp

def load(name):
    list_path = [os.getcwd(),'env', name]
    pathname = os.path.join(*list_path)
    print(pathname)
    return imp.load_source('', pathname)

def make_env(scenario_name, args):
    scenario = load(scenario_name + ".py").Scenario()
    #scenario = simple_tag.Scenario()
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




def train(args):
    env = make_env(args.scenario, args)
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
    path_length = 0
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    for n in range(num_steps):
        #print(f'Episode {n} begins')
        action_n = [agent.act(torch.tensor(o)) for agent, o in zip(trainers, obs_n)]
        next_obs_n, rew_n, done_n, info_n = env.step(action_n)
        done = all(done_n)
        for i, agent in enumerate(trainers):
            episode_rewards[-1] += rew_n[i]
            agent_rewards[i][-1] += rew_n[i]
            agent.experience(obs_n[i], action_n[i], rew_n[i], next_obs_n[i])
        path_length += 1
        obs_n = next_obs_n
        terminal = (path_length >= args.max_path_length)
        if done or terminal:
            print(f"saving this episode's reward...")
            obs_n = env.reset()
            path_length = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
        # update
        for i, agent in enumerate(trainers):
            current_agent = i
            agent.update(trainers, batch_size)

        # save model, display training output
        if (done or terminal) and len(episode_rewards) % args.evaluate_freq == 0:  # every time this many episodes are complete
            if num_adv == 0:
                print("steps: {}, episodes: {}, episode reward: {}, time: {}".format(
                    n, len(episode_rewards), np.mean(episode_rewards[-args.evaluate_freq:])))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}".format(
                    n, len(episode_rewards), np.mean(episode_rewards[-args.evaluate_freq:]), 
                    [np.mean(rew[-args.evaluate_freq:]) for rew in agent_rewards]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_type", type=str, default="mpe", help="name of scenario")
    parser.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--num_adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--batch_size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--train_steps_per_episode", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--evaluate_freq", type=int, default=100)

    args = parser.parse_args()
    train(args)