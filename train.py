import pettingzoo.mpe as mpe
import argparse
from trainer.MADDPG import MADDPG

def make_env(scenario_name, args):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_space, args):
    trainers = []
    for i in range(num_adversaries):
        trainers.append(MADDPG(obs_space, env.action_spaces, args))
    for i in range(num_adversaries, env.num_agents):
        trainers.append(MADDPG(obs_space, env.action_spaces, args))
    return trainers


def evaluate(args):
    return


def train(args):
    if args.scenario_type == 'mpe':
        env = mpe.simple_tag_v2.env()  # Todo
    else:
        env = make_env(args.scenario, args)
    env.reset()
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.num_agents)]  # individual agent reward

    obs_shape_n = [env.observation_spaces['agent_0'].shape for i in range(env.num_agents)] # obs shape in list
    num_adversaries = min(env.num_agents, args.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, args)
    num_steps_collected = 0
    for t in range(args.train_step):
        obs = [[] for _ in env.num_agents]
        actions = [[] for _ in env.num_agents]
        rews = [[] for _ in env.num_agents]
        new_obs = [[] for _ in env.num_agents]
        done = [[] for _ in env.num_agents]
        path_length = 0
        while path_length < args.max_path_length:
            action_n = [agent.act(o)[0] for agent, o in zip(trainers, obs)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            done = all(done_n)
            # save replay buffer
            for i, agent in enumerate(trainers):
                obs[i].append(obs_n[i])
                actions[i].append(action_n[i])
                rews[i].append(rew_n[i])
                new_obs[i].append(new_obs_n[i])
                done[i].append(done_n[i])
            path_length += 1
            if done:
                break
            obs_n = new_obs_n

        for i, agent in enumerate(trainers):
            agent.experience(obs[i], actions[i], rews[i], new_obs[i], done[i])
        num_steps_collected += len(obs[0])

        for i, rew in enumerate(rews):
            episode_rewards[-1] += np.sum(rew)
            agent_rewards[i][-1] += np.sum(rew)
                    
        # update
        for agent in trainers:
            agent.update(trainers, args.batch_size)


        # save model, display training output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_type", type=str, default="mpe", help="name of scenario")
    parser.add_argument("--scenario", type=str, default="simple_v2", help="name of the scenario script")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--batch_size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--train_step", type=int, default=1000)
    parser.add_argument("--max_path_length", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=1000)
    
    args = parser.parse_args()
    train(args)