import gym
from gym.utils import seeding
import logging
import numpy as np
import networkx as ne  # 导入建网络模型包，命名ne
import matplotlib.pyplot as mp

logger = logging.getLogger(__name__)


class SWEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        # 小世界网络生成基本设置
        self.NETWORK_SIZE = 30
        self.K = 15
        self.reconnect_p = 0.5
        self.cm = None  # 邻接矩阵
        self.cm_weight = None
        self.ws = None  # 图 networkx
        self.ps = None  # 图框架
        self.max_weight = 5
        self.change_weight = 1.5
        self.create_weight = np.random.rand

        # 状态空间设置
        self.observation_space = list(range(30))  # 状态空间
        self.__terminal_finial = 30  # 终止状态为字典格式

        # 状态转移
        pass

        # 环境基本设置
        self.__gamma = 0.8  # 折扣因子
        self.viewer = None  # 环境可视化
        self.__state = None  # 当前状态
        self.seed()  # 随机种子

    def _reward(self, state, action):
        """
        回报
        :param state:
        :param action
        :return:
        """
        r = 0.0
        step_num = state[0]
        local_point = state[1]
        cm = state[2]
        cm_weight = [3]
        # 终止
        if cm[local_point, action] == 0 or step_num > 30:
            return -10000
        # 到达
        if self.__terminal_finial == action:
            return 1000
        # 经过
        return -cm_weight[local_point, action]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def transform(self, state, action):
        """
        状态转换
        :param state:
        :param action:
        :return:
        """
        step_num = state[0]
        local_point = state[1]
        cm = state[2]
        cm_weight = [3]

        # 计算回报
        r = self._reward(state, action)

        # 判断是否终止
        if cm[local_point, action] == 0 or step_num > 30 or self.__terminal_finial == action:
            return state, self._reward(state, action), True, {}
        next_state = [[], [], [], []]
        # 状态转移
        next_state[0] = state[0] + 1
        next_state[1] = action
        next_state[2] = cm
        # 浮动
        cm_weight = cm_weight + (self.create_weight(self.cm[0], self.cm[1]) - 0.5) * self.change_weight
        next_state[3] = cm_weight
        # 判断是否终止
        is_terminal = False

        return next_state, r, is_terminal, {}

    def step(self, action):
        """
        交互
        :param action: 动作值
        :return: 下一个状态，回报，是否停止，调试信息
        """
        state = self.__state
        next_state, r, is_terminal, _ = self.transform(state, action)
        self.__state = next_state
        return next_state, r, is_terminal, {}

    def reset(self):
        """
        重置环境
        :return: [步数，位置，邻接矩阵，带权矩阵]
        """
        # 重新生成小世界网络
        self.ws = ne.watts_strogatz_graph(self.NETWORK_SIZE, self.K, self.reconnect_p)
        self.ps = ne.circular_layout(self.ws)  # 布置框架
        # 可视化
        # self.viewer = ne.draw(self.ws, self.ps, with_labels=False, node_size=self.NETWORK_SIZE)
        self.cm = np.array(ne.adjacency_matrix(self.ws).todense())  # 邻接矩阵
        self.cm_weight = self.cm * self.create_weight(self.cm[0], self.cm[1]) * self.max_weight  # 邻接

        # 设置起点
        self.__state = [0, 0, self.cm, self.cm_weight]  # 步数，位置，邻接矩阵，带权矩阵
        return self.__state  # 步数，位置，邻接矩阵，带权矩阵

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                mp.show()


if __name__ == '__main__':
    env = SWEnv()
    env.reset()
    env.render(closs=True)
