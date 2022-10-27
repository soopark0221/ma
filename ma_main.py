from env.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch,os

from trainer.maddpg_agent import MADDPG
from trainer.bootmaddpg import BootMADDPG
from trainer.MA import SWAGMA
from trainer.swag import SWAG

from trainer.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state

from trainer.utils import *
from copy import deepcopy
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def get_trainers(env, num_adversaries, obs_space, action_space, args):
    trainers = []
    for i in range(num_adversaries):
        trainers.append(SWAGMA(i, obs_space, action_space, args))
    for i in range(num_adversaries, env.n):
        trainers.append(SWAGMA(i, obs_space, action_space, args))
    return trainers


def main(args):

    env = make_env(args.scenario)
    print(f'configs: scenario {args.scenario}, algo {args.algo}, seed {args.seed}')
    torch.manual_seed(args.seed)

    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir+args.scenario+"_seed"+str(args.seed))
    
    if args.algo == "swagma":

        num_adv=args.num_adv
        obs_space=env.observation_space
        action_space=env.action_space
        
        model=get_trainers(env, num_adv, obs_space, action_space, args)

        # train
        path_length = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        obs_n = env.reset()

        for n in range(args.max_steps):

            action_n = [agent.act(torch.tensor(o)) for agent, o in zip(model, obs_n)]
            action_n = np.array(action_n)
            #action_n=torch.tensor(action_n).data.cpu().numpy()
            next_obs_n, rew_n, done_n, info_n = env.step(action_n)
            done = all(done_n)
            for i, agent in enumerate(model):
                episode_rewards[-1] += rew_n[i]
                agent_rewards[i][-1] += rew_n[i]
                agent.experience(obs_n[i], action_n[i], rew_n[i], next_obs_n[i])
            path_length += 1
            obs_n = next_obs_n
            terminal = (path_length >= args.perepisode_length)
            if done or terminal:
                #print(f"end of episode, total step: {n}")
                for i, agent in enumerate(model):
                    agent.update(model)
                obs_n = env.reset()
                path_length = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)


            '''
            if n % args.collect_freq == 0:
                for i, agent in enumerate(model):
                    agent.collect_params()

            if n % args.sample_freq == 0:
                for i, agent in enumerate(model):
                    agent.sample_params()
            '''
            # save model, display training output
            if (done or terminal) and len(episode_rewards) % args.evaluate_freq == 0:  # every time this many episodes are complete
                agent_episode_reward = [np.sum(rew[-args.evaluate_freq:]) for rew in agent_rewards]
                if num_adv > 0 :
                    adv_episode_reward = np.sum(agent_episode_reward[:num_adv])
                    ag_episode_reward = np.sum(agent_episode_reward[-1])
                else:
                    adv_episode_reward = 0
                    ag_episode_reward = np.sum(agent_episode_reward)

                if num_adv == 0:
                    print("steps: {}, episodes: {}, episode reward: {}, episode reward sum: {}".format(
                        n, len(episode_rewards), np.mean(episode_rewards[-args.evaluate_freq:]), np.sum(episode_rewards[-args.evaluate_freq:])))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, episode reward sum: {}, agent episode reward: {}".format(
                        n, len(episode_rewards), np.mean(episode_rewards[-args.evaluate_freq:]), np.sum(episode_rewards[-args.evaluate_freq:]),
                        [np.mean(rew[-args.evaluate_freq:]) for rew in agent_rewards]))

                if args.tensorboard:
                    writer.add_scalar(tag='agent/total_reward', global_step=n, scalar_value=np.sum(episode_rewards[-args.evaluate_freq:]))
                    writer.add_scalar(tag='agent/adv_reward', global_step=n, scalar_value=adv_episode_reward)
                    writer.add_scalar(tag='agent/agent_reward', global_step=n, scalar_value=ag_episode_reward)



    else:

        n_agents = env.n
        n_actions = env.world.dim_p*2+1 
        # XXX: from environment.py, discrete action space
        
        # env = ActionNormalizedEnv(env)
        # env = ObsEnv(env)
        n_states = env.observation_space[0].shape[0]
        if args.algo == "maddpg" or args.algo == 'swag_maddpg':
            model = MADDPG(n_states, n_actions, n_agents, args)

        if args.algo == "bootmaddpg":
            model = BootMADDPG(n_states, n_actions, n_agents, args)
    

        print(model)
        model.load_model()

        episode = 0
        total_step = 0

        while episode < args.model_episode:

            state = env.reset()

            episode += 1
            step = 0
            accum_reward = 0
            adv_epi_reward=0
            agent_epi_reward=0

            while True:

                if args.mode == "train":
                    action = model.choose_action(state, noisy=True)
                    next_state, reward, done, info = env.step(action)

                    step += 1
                    total_step += 1
                    reward = np.array(reward)
                    accum_reward += np.sum(reward)

                    if args.num_adv > 0:
                        adv_epi_reward += np.sum(reward[0:args.num_adv])   # XXX: for num_adv=3
                        agent_epi_reward += reward[-1]
                    else:
                        agent_epi_reward += np.sum(reward)

                    obs = torch.from_numpy(np.stack(state)).float().to(device)
                    obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)

                    next_obs = obs_
                    #if step != args.perepisode_length - 1:
                    #    next_obs = obs_
                    #else:
                    #    next_obs = None
                    rw_tensor = torch.FloatTensor(reward).to(device)
                    ac_tensor = torch.FloatTensor(action).to(device)
                    
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                    obs = next_obs

                    state = next_state
                    if args.display and episode >= 300:
                        env.render()
                    if args.perepisode_length < step or (True in done):
                        model.prep_training()
                        c_loss, a_loss = model.update(episode)  # XXX: update per episode or step?

                        if episode % args.print_interval == 0: 
                            print("[Episode %05d] reward %6.4f adv_reward %6.4f agent_reward %6.4f" % (episode, accum_reward, adv_epi_reward, agent_epi_reward))
                            
                            if args.tensorboard:
                                writer.add_scalar(tag='agent/total_reward', global_step=episode, scalar_value=accum_reward)
                                writer.add_scalar(tag='agent/adv_reward', global_step=episode, scalar_value=adv_epi_reward)
                                writer.add_scalar(tag='agent/agent_reward', global_step=episode, scalar_value=agent_epi_reward)
                            
                                if c_loss and a_loss:
                                    writer.add_scalar('agent/actor_loss', global_step=episode,
                                                    scalar_value= a_loss)
                                    writer.add_scalar('agent/critic_loss', global_step=episode,
                                                    scalar_value= c_loss)  
                            if c_loss and a_loss:
                                print(" a_loss %3.4f c_loss %3.4f" % (a_loss, c_loss), end='')


                            if episode % args.save_interval == 0 and args.mode == "train":
                                model.save_model(episode)
                        env.reset()
                        # model.reset()
                        break
                
                elif args.mode == "eval":
                    action = model.choose_action(state, noisy=False)
                    next_state, reward, done, info = env.step(action)
                    step += 1
                    total_step += 1
                    state = next_state
                    reward = np.array(reward)
                    import time
                    time.sleep(0.02)
                    env.render()

                    accum_reward += np.sum(reward)
                    adv_epi_reward+=np.sum(reward[0:2])
                    agent_epi_reward=reward[3]

                    if args.perepisode_length < step or (True in done):
                        print("[Episode %05d] reward %6.4f " % (episode, accum_reward))
                        env.reset()
                        break
            
            if args.algo == 'swag_maddpg':
                if episode % args.collect_freq == 0:
                    model.collect_params()
                if episode % args.sample_freq == 0:
                    model.sample_params()

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_tag", type=str)
    parser.add_argument('--max_steps', default=int(1e8), type=int)
    parser.add_argument('--algo', default="bootmaddpg", type=str, help="maddpg/bootmaddpg/swagma")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--num_adv', type=int, default=3)
    parser.add_argument('--perepisode_length', default=100, type=int)
    parser.add_argument('--memory_length', default=int(5*1e5), type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--n_ensemble',default=5, type=int)
    parser.add_argument('--a_lr', default=0.001, type=float)
    parser.add_argument('--c_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=100000, type=int)
    parser.add_argument("--print_interval", default=1000, type=int)
    parser.add_argument("--model_episode", default=100000, type=int)
    parser.add_argument('--episode_before_train', default=1000, type=int)
    parser.add_argument('--steps_before_train', default=10000, type=int, help="for swagma, start update after this number of steps")
    parser.add_argument("--evaluate_freq", type=int, default=10)
    parser.add_argument("--collect_freq", type=int, default=100)
    parser.add_argument("--sample_freq", type=int, default=5000)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%m%d_%H%M'))
    parser.add_argument('--display', action="store_true", default=False)

    args = parser.parse_args()
    main(args)
