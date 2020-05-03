from agent import Agent
from environment import SWEnv
import logging
import datetime
import torch
from pathlib import Path


def get_logger(log_dir, file_filter="DEBUG", name=__name__, on_screen=True):
    """
    get logger
    :param name: log name
    :param log_dir: log dir
    :param file_filter: DEBUG or INFO
    :param on_screen : should print log？ T or F
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, name))
    assert (file_filter == "DEBUG" or file_filter == "INFO")
    if file_filter == "DEBUG":
        file_handler.setLevel(logging.DEBUG)
    elif file_filter == "INFO":
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if on_screen:  # print log on console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


def main(iteration_num=1000,
         iteration_grah=100,
         checkpoint_freq=2,
         gpu_list='0',
         log_dir=None,
         actor_pretrain='',
         critic_pretrain=''):
    """GET TIME"""
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    """CREATE DIR"""
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    """BEGIN LOGGING"""
    logger = get_logger(log_dir, name="agent_0")
    logger.info("EXPERIMENT TIME:" + timestr)
    # 创建环境
    env = SWEnv()
    # 创建个体
    agent = Agent()
    # 初始化网络
    if not gpu_list == '':
        agent.actor_net = agent.actor_net.cuda()
        agent.critic_net = agent.critic_net.cuda()
        agent.actor_net_now = agent.actor_net_now.cuda()
        agent.critic_net_now = agent.critic_net_now.cuda()
        agent.actor_net_loss.cuda()
        agent.critic_net_loss.cuda()
    if not actor_pretrain == '':
        agent.actor_net.load_state_dict(torch.load(actor_pretrain))
        agent.actor_net_now.load_state_dict(torch.load(actor_pretrain))
    if not critic_pretrain == '':
        agent.critic_net.load_state_dict(torch.load(actor_pretrain))
        agent.critic_net_now.load_state_dict(torch.load(actor_pretrain))
    agent.actor_net.train()
    agent.critic_net.train()
    agent.actor_net_now.train()
    agent.critic_net_now.train()
    # 优化器
    agent.actor_optimizer = torch.optim.Adam(
        agent.actor_net_now.parameters(),
        lr=0.001,  # 学习率
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    actor_scheduler = torch.optim.lr_scheduler.StepLR(agent.actor_optimizer, step_size=20, gamma=0.7)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(agent.critic_optimizer, step_size=20, gamma=0.7)

    for it in range(iteration_num):
        # 生成图
        env.change_graph()
        for itg in iteration_grah:
            # 重置环境
            state = env.change_graph()
            state = agent.state2actor_tensor(state)  # 转化为tensor
            if gpu_list == '':
                state = state.cuda()
            is_terminal = True
            # 清空ep
            agent.experience_playback = {'state': [],
                                         'action': [],
                                         'reward': [],
                                         'next_state': [],
                                         'is_terminal': []}
            # 迭代训练
            actor_scheduler.step()
            critic_scheduler.step()
            while not is_terminal:
                # 动作网络
                a_output = agent.actor_net_now(state)
                # 执行动作
                a = a_output.data.max(1)[1]
                next_state, r, is_terminal, _ = env.step(int(a[0]))
                next_state = agent.state2actor_tensor(next_state)
                r = torch.Tensor([r])

                if is_terminal:
                    is_terminal_bit = torch.Tensor([0])
                else:
                    is_terminal_bit = torch.Tensor([1])
                if gpu_list == '':
                    next_state = next_state.cuda()
                    r = r.cuda()
                    is_terminal_bit = is_terminal_bit.cuda()
                # 经验回放
                agent.experience_playback['state'].append(state)
                agent.experience_playback['action'].append(a_output)
                agent.experience_playback['reward'].append(r)
                agent.experience_playback['next_state'].append(next_state)
                agent.experience_playback['is_terminal'].append(is_terminal_bit)
                state = next_state
                # 更新当前critic网络
                agent.critic_learn(agent.batch_size, gpu_list=gpu_list)
                # 更新当前actor网络
                agent.actor_learn(agent.batch_size, gpu_list=gpu_list)
            # 目标网络参数更新
            if itg % agent.actor_updata_freq == 0:  # 目标动作网络更新
                agent.update_now_net2target_net(agent.actor_net_now, agent.actor_net, agent.tau_actor)
            if itg % agent.critic_updata_freq == 0:  # 目标动作网络更新
                agent.update_now_net2target_net(agent.critic_net_now, agent.critic_net, agent.tau_critic)
            # 网络参数保存
            if itg % checkpoint_freq == 0:
                logger.info('Save model...')
                actor_savepath = str(checkpoints_dir) + '/actor_model_iter' + str(it) + '.pth'
                critic_savepath = str(checkpoints_dir) + '/critic_model_iter' + str(it) + '.pth'
                logger.info('Actor Net Saving at %s' % actor_savepath)
                logger.info('Critic Net Saving at %s' % critic_savepath)
                torch.save(agent.actor_net.state_dict(), actor_savepath)
                torch.save(agent.critic_net.state_dict(), critic_savepath)


if __name__ == '__main__':
    main()
