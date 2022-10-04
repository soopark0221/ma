from env.make_env import make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch,os
import random

from trainer.maddpg import MADDPG
from trainer.bootmaddpg import BootMADDPGa, BootMADDPGc
from trainer.swag import SWAG

from trainer.normalized_env import ActionNormalizedEnv, ObsEnv, reward_from_state

from trainer.utils import *
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main(args):

    env = make_env(args.scenario)
    print(f'configs: scenario {args.scenario}, algo {args.algo}, seed {args.seed}')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.log_dir+args.scenario+"_seed"+str(args.seed))
    
    n_agents = env.n
    n_actions = env.world.dim_p*2+1 
    # XXX: from environment.py, if discrete action space
    
    n_states = env.observation_space[0].shape[0]
    if args.algo == "maddpg" or args.algo == 'swag_maddpg':
        model = MADDPG(n_states, n_actions, n_agents, args)

    if args.algo == "bootmaddpgc":
        model = BootMADDPGc(n_states, n_actions, n_agents, args)

    if args.algo == "bootmaddpga":
        model = BootMADDPGa(n_states, n_actions, n_agents, args)


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
        if args.algo=='bootmaddpga':
            model.epi_actors=np.random.randint(model.n_ensemble,size=n_agents)
        while True:

            if args.mode == "train":
                
                action = model.choose_action(state, noisy=True)

                next_state, reward, done, info = env.step(action)

                step += 1
                total_step += 1
                reward = np.array(reward)

                accum_reward += np.sum(reward)
                if args.num_adv > 0:
                    num_adv=int(args.num_adv)
                    adv_epi_reward += np.sum(reward[0:num_adv])   
                    agent_epi_reward += np.sum(reward[num_adv:])
                

                obs = torch.from_numpy(np.stack(state)).float().to(device)
                obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                
                next_obs = obs_

                if step != args.perepisode_length - 1:		                   
                    next_obs = obs_		              
                else:		       
                    next_obs = None
                rw_tensor = torch.FloatTensor(reward).to(device)
                ac_tensor = torch.FloatTensor(action).to(device)
                
                if hasattr(model, 'epi_actors'):
                    epi_actors=model.epi_actors
                else:
                    epi_actors=None
                model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, epi_actors)
                obs = next_obs

                state = next_state
                

                if args.perepisode_length < step or (True in done):
                    c_loss, a_loss = model.update(episode)  
                    print("[Episode %05d] reward %6.4f adv_reward %6.4f agent_reward %6.4f" % (episode, accum_reward, adv_epi_reward, agent_epi_reward))
                    
                    if args.tensorboard:
                        writer.add_scalar(tag='agent/total_reward', global_step=episode, scalar_value=accum_reward)
                        if args.num_adv>0:
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
    # parser.add_argument('--max_steps', default=int(1e8), type=int)
    parser.add_argument('--algo', default="maddpg", type=str, help="maddpg/bootmaddpgc,a/swag_maddpg")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--num_adv', type=int, default=3, help="manually, tag=3, spread=0 etc")
    parser.add_argument('--perepisode_length', default=100, type=int)
    parser.add_argument('--memory_length', default=int(5*1e5), type=int)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--n_ensemble',default=5, type=int)
    parser.add_argument('--a_lr', default=0.0013, type=float)
    parser.add_argument('--c_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--model_episode", default=100000, type=int)
    parser.add_argument('--episode_before_train', default=100, type=int)
    # parser.add_argument("--evaluate_freq", type=int, default=10, help="update freq")
    parser.add_argument("--collect_freq", type=int, default=100, help="store freq")
    parser.add_argument("--sample_freq", type=int, default=500)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%m%d_%H%M'))

    args = parser.parse_args()
    main(args)
