import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from src.module.bandit.utils.numba_helper import numba_dot, numba_identity, numba_sqrt, \
    numba_linalg_inv, numba_outer, numba_transpose, \
    numba_zeros
from utils.logger import logger


class NumbaLinUBC():
    '''
    Numba optimized LinUBC object

    :param arms: list of all available arms
    :type arms: list of string
    :param context_size: context size which is equivalent to number of features
    :type context_size: int
    :param alpha: number to control the balance between exploration and exploitation
    :type alpha: float
    :param logger: logger object
    :type logger: python logger or Logger object
    '''

    def __init__(self, arms: List[str], context_size: int, alpha: float = 0.001, logger=logger):

        self.arms = arms
        self.context_size = context_size
        self.alpha = alpha
        self.logger = logger

        self.init_bandit(arms, context_size)

        # store CTR at every time step for Cumulative Take-Rate replay
        self.CTR = {}
        self.correct_cnt = [0]  # initialize

    def init_bandit(self, arms, feature_size):
        '''Method for initializing bandit model

        :param arms: all unique arms
        :type arms: list of string
        :param feature_size: number of context features
        :type feature_size: int
        '''

        n_arms = len(arms)
        self.A = numba_zeros((n_arms, feature_size, feature_size))
        self.A_inv = numba_zeros((n_arms, feature_size, feature_size))
        self.b = numba_zeros((n_arms, feature_size, 1))
        self.theta = numba_zeros((n_arms, feature_size, 1))

        self.arm2index = {}
        self.index2arm = {}

        self.logger.info('Initialize model')

        for i, arm in enumerate(arms):
            self.arm2index[arm] = i
            self.index2arm[i] = arm

            self.A[i] = numba_identity(feature_size)
            self.A_inv[i] = numba_identity(feature_size)
            self.b[i] = numba_zeros((feature_size, 1))
            self.theta[i] = numba_zeros((feature_size, 1))

        self.ucb_mean = numba_zeros((n_arms, 1, 1))  # the ucb calculation shape is (n_arms, 1, 1)
        self.arm_reward_mean = numba_zeros(n_arms)

    def train(self, data: pd.DataFrame, epochs: int, n_visits: int, shuffle: bool = True,
              arm_col: str = 'uuid', candidate_col: str = 'uuids', reward_col: str = 'reward', warmup_iters: int = 0, **kwargs):
        '''Train method

        :param data: training data
        :type data: pd.DataFrame
        :param epochs: number of epochs to train the bandit model
        :type epochs: int
        :param n_visits: number of iterations within one epoch. n_visits = len(dataset) if n_visits is not set
        :type n_visits: int
        :param shuffle: whether to random shuffle the training data
        :type shuffle: bool
        :param arm_col: column name of arm
        :type arm_col: string
        :param candidate_col: column holds list of candidates including positive and negative samples
        :type candidate_col: string
        :param reward_col: column name of reward
        :type reward_col: string
        :param warmup_iters: number of iterations for warmup. if set to -1, warmup_iters = n_visits
        :type warmup_iters: int
        '''
        # training data must with format of [[arm_id, feature_1, feature_2, ...]]
        n_visits = n_visits if n_visits else len(data)

        self.logger.info(f'Training data: \n{data.head(3)}')

        if shuffle:
            data = data.sample(frac=1)
            self.logger.info(f'Training data after shuffle: \n{data.head(3)}')

        data_triplet = list(zip(data[arm_col].values,
                                data[candidate_col].values,
                                data[reward_col].values,
                                data.drop([arm_col, candidate_col, reward_col], axis=1).values))

        data_triplet = list(map(lambda x: (x[0], x[1], x[2], x[3].reshape((self.context_size, 1))),
                                data_triplet))

        self.logger.info(f'Training data triplet: \n{data_triplet[:3]}')

        self.logger.info(f'Start training for {epochs} epochs and {n_visits} iterations per epochs')

        self.cumulative_reward = 0
        self.ctr_num = 0
        self.ctr_den = 0
        self.CTR = {}
        self.visit_time = 1

        if warmup_iters == -1:
            warmup_iters = n_visits
            self.logger.info(f'Do warmup for {warmup_iters} iterations')
        elif warmup_iters:
            self.logger.info(f'Do warmup for {warmup_iters} iterations')

        for epoch_step in range(epochs):

            pbar = tqdm(range(n_visits))
            for i in pbar:
                pbar.set_description(f'Epoch: {epoch_step}')
                cur_arm, cur_candidates, cur_reward, cur_feature = data_triplet[i]
                cur_feature_t = numba_transpose(cur_feature, None)
                index = [self.arm2index[arm] for arm in cur_candidates]

                cur_ucb = np.matmul(numba_transpose(self.theta[index], (0, 2, 1)), cur_feature) \
                    + self.alpha * numba_sqrt(np.matmul(cur_feature_t, np.dot(self.A_inv[index], cur_feature)))

                max_idx = index[np.argmax(cur_ucb)]
                pred_arm = self.index2arm[max_idx]

                # If the prediction matches the current arm, update values
                if self.visit_time < warmup_iters:
                    pred_arm = cur_arm
                    max_idx = self.arm2index[pred_arm]

                if cur_arm == pred_arm:
                    self.ucb_mean[max_idx] += cur_ucb[np.argmax(cur_ucb)]

                    self.A[max_idx] += numba_outer(cur_feature, cur_feature)
                    self.A_inv[max_idx] = numba_linalg_inv(self.A[max_idx])

                    self.b[max_idx] += cur_reward * cur_feature
                    self.theta[max_idx] = numba_dot(self.A_inv[max_idx], self.b[max_idx])

                    self.cumulative_reward += cur_reward

                    cur_ctr, self.ctr_num, self.ctr_den = self.calc_ctr(cur_reward, self.ctr_num, self.ctr_den)
                    self.CTR[self.visit_time] = cur_ctr
                    self.correct_cnt.append(self.correct_cnt[-1] + 1)
                else:
                    self.correct_cnt.append(self.correct_cnt[-1])

                self.visit_time += 1

    def calc_ctr(self, reward: int, ctr_num: int, ctr_den: int):
        '''Method to calculate the cumulative take rate. 
        Keeps a count of CTR and updates the numerator and denominator on every correct call

        :param reward: reward value
        :type reward: int
        :param reward: ctr numerator
        :type reward: int
        :param reward: ctr denominator
        :type reward: int
        :return: ctr, ctr numerator and ctr denominator
        :rtype: sequence of int
        '''

        ctr_num = ctr_num + reward
        ctr_den = ctr_den + 1

        ctr = ctr_num/ctr_den

        return ctr, ctr_num, ctr_den
