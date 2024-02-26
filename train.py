import argparse
import glob
import os
import shutil
from collections import deque

import colorama
import gymnasium
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

import agents
import env_wrapper
from replay_buffer import ReplayBuffer
from sub_models.world_models import WorldModel
from utils import Logger, load_config, seed_np_torch


def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(
        env_name, full_action_space=False, render_mode="rgb_array", frameskip=1
    )
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)

    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(
    num_episodes,
    env_name,
    num_envs,
    image_size,
    world_model: WorldModel,
    agent: agents.ActorCriticAgent,
    seed,
):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(
                    torch.cat(list(context_obs), dim=1)
                )
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                (
                    prior_flattened_sample,
                    last_dist_feat,
                ) = world_model.calc_last_dist_feat(
                    context_latent, model_context_action
                )
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False,
                )

        context_obs.append(
            rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W") / 255
        )
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episodes:
                        print(
                            "Mean reward: "
                            + colorama.Fore.YELLOW
                            + f"{np.mean(final_rewards)}"
                            + colorama.Style.RESET_ALL
                        )
                        vec_env.close()
                        return np.mean(final_rewards)

        # update context_obs and context_action
        context_obs.append(
            rearrange(torch.Tensor(obs).cuda(), "B H W C -> B 1 C H W") / 255
        )
        context_action.append(action)

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part


def train_world_model_step(
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    batch_size,
    demonstration_batch_size,
    batch_length,
    logger,
):
    obs, action, reward, termination = replay_buffer.sample(
        batch_size, demonstration_batch_size, batch_length
    )
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: agents.ActorCriticAgent,
    imagine_batch_size,
    imagine_demonstration_batch_size,
    imagine_context_length,
    imagine_batch_length,
    log_video,
    logger,
):
    """
    Sample context from replay buffer, then imagine data with world model and agent
    """
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length
    )
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent,
        sample_obs,
        sample_action,
        imagine_batch_size=imagine_batch_size + imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger,
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_eval_world_model_agent(
    env_name,
    max_steps,
    num_envs,
    image_size,
    replay_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: agents.ActorCriticAgent,
    train_dynamics_every_steps,
    train_agent_every_steps,
    batch_size,
    demonstration_batch_size,
    batch_length,
    imagine_batch_size,
    imagine_demonstration_batch_size,
    imagine_context_length,
    imagine_batch_length,
    save_every_steps,
    eval_every_steps,
    eval_num_envs,
    eval_num_episodess,
    seed,
    logger,
):
    ckptdir = f"ckpt/{env_name}/{seed}/"
    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print(
        "Current env: "
        + colorama.Fore.YELLOW
        + f"{env_name}"
        + colorama.Style.RESET_ALL
    )

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train and eval
    for total_steps in tqdm(
        range(logger.step, max_steps // num_envs),
        initial=logger.step,
        total=max_steps // num_envs,
    ):
        logger.step = total_steps
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(
                        torch.cat(list(context_obs), dim=1)
                    )
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    (
                        prior_flattened_sample,
                        last_dist_feat,
                    ) = world_model.calc_last_dist_feat(
                        context_latent, model_context_action
                    )
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False,
                    )

            context_obs.append(
                rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")
                / 255
            )
            context_action.append(action)
        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(
            current_obs, action, reward, np.logical_or(done, info["life_loss"])
        )

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(
                        f"sample/{env_name}_episode_steps",
                        current_info["episode_frame_number"][i] // 4,
                    )  # framskip=4
                    logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        if (
            replay_buffer.ready()
            and total_steps % (train_dynamics_every_steps // num_envs) == 0
        ):
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger,
            )
        # <<< train world model part

        # train agent part >>>
        if (
            replay_buffer.ready()
            and total_steps % (train_agent_every_steps // num_envs) == 0
            and total_steps * num_envs >= 0
        ):
            if total_steps % (save_every_steps // num_envs) == 0:
                log_video = True
            else:
                log_video = False

            (
                imagine_latent,
                agent_action,
                agent_logprob,
                agent_value,
                imagine_reward,
                imagine_termination,
            ) = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger,
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
            )
        # <<< train agent part

        # evaluate agent
        if total_steps % (eval_every_steps // num_envs) == 0:
            print(
                colorama.Fore.GREEN
                + f"Evaluating at total steps {total_steps}"
                + colorama.Style.RESET_ALL
            )
            mean_rewards = eval_episodes(
                num_episodes=eval_num_episodess,
                env_name=env_name,
                num_envs=eval_num_envs,
                image_size=image_size,
                world_model=world_model,
                agent=agent,
                seed=seed,
            )
            logger.log(f"eval/{env_name}_mean_reward", mean_rewards)
            logger.write()

        # save model per episode
        if total_steps % (save_every_steps // num_envs) == 0:
            print(
                colorama.Fore.GREEN
                + f"Saving model at total steps {total_steps}"
                + colorama.Style.RESET_ALL
            )
            torch.save(
                world_model.state_dict(), ckptdir + f"world_model_{total_steps}.pth"
            )
            torch.save(agent.state_dict(), ckptdir + f"agent_{total_steps}.pth")


def build_world_model(conf, action_dim):
    model = WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        use_amp=conf.BasicSettings.UseAmp,
    ).cuda()
    return torch.compile(model)


def build_agent(conf, action_dim):
    agent = agents.ActorCriticAgent(
        feat_dim=32 * 32 + conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
        use_amp=conf.BasicSettings.UseAmp,
    ).cuda()
    return torch.compile(agent)


if __name__ == "__main__":
    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-trajectory_path", type=str)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logdir = f"runs/{args.n}/{args.seed}/"
    os.makedirs(logdir, exist_ok=True)
    ckptdir = f"ckpt/{args.n}/{args.seed}/"
    os.makedirs(ckptdir, exist_ok=True)
    logger = Logger(logdir=logdir, step=0)
    # copy config file
    shutil.copy(args.config_path, logdir + "config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(
            args.env_name, conf.BasicSettings.ImageSize, seed=0
        )
        action_dim = dummy_env.action_space.n

        # build world model and agent
        world_model = build_world_model(conf, action_dim)
        agent = build_agent(conf, action_dim)

        # load world model and agent from checkpoint if present
        paths = glob.glob(ckptdir + "world_model_*.pth")
        if paths:
            steps = [int(path.split("_")[-1].split(".")[0]) for path in paths]
            last_step = max(steps)
            world_model_path = ckptdir + f"world_model_{last_step}.pth"
            agent_path = ckptdir + f"agent_{last_step}.pth"
            print(
                colorama.Fore.MAGENTA
                + f"loading world model from {world_model_path}"
                + colorama.Style.RESET_ALL
            )
            world_model.load_state_dict(torch.load(world_model_path))
            print(
                colorama.Fore.MAGENTA
                + f"loading agent from {agent_path}"
                + colorama.Style.RESET_ALL
            )
            agent.load_state_dict(torch.load(agent_path))
            logger.step = last_step

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(
                colorama.Fore.MAGENTA
                + f"loading demonstration trajectory from {args.trajectory_path}"
                + colorama.Style.RESET_ALL
            )
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train and eval
        joint_train_eval_world_model_agent(
            env_name=args.env_name,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize
            if conf.JointTrainAgent.UseDemonstration
            else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize
            if conf.JointTrainAgent.UseDemonstration
            else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            eval_every_steps=conf.JointTrainAgent.EvalEverySteps,
            eval_num_envs=conf.JointTrainAgent.EvalNumEnvs,
            eval_num_episodess=conf.JointTrainAgent.EvalNumEpisodes,
            seed=args.seed,
            logger=logger,
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
