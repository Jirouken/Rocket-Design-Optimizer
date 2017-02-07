# -*- coding: utf-8 -*-

import numpy as np
from numpy import random as rand
from Simulator import Simulator
from matplotlib import pyplot as plt


class GeneticAlgorithm(object):
    """遺伝的アルゴリズム"""

    def __init__(self):
        """初期化"""

        """定数"""
        # 染色体数
        self.N = 4
        # 個体数
        self.M = 20
        # エリート選択個数
        self.n_elite = 4
        # ルーレット選択個数
        self.n_roulette = 2
        # 突然変異を起こす確率
        self.p_mutation = 0.05

        """変数"""
        # 第g世代評価
        self.value_g = []
        # 第g世代評価降順インデックス
        self.val_sorted_index = []
        # 最大評価値
        self.values = []
        # 選択された遺伝子のインデックス
        self.selected_index = []
        # 次世代遺伝子
        self.next_generation = []

        """初期値"""
        self.X = []
        for i in xrange(self.M):
            # エンジンの種類
            e = rand.randint(5)
            # 機体直径
            d = 0.05 + rand.rand() * 0.5
            # 機体長さ
            L = 0.7 + rand.rand() * 7
            # 錘質量
            m_w = rand.rand()
            # 遺伝子配列
            x = [e, d, L, m_w]
            self.X.append(x)
        self.X = np.array(self.X)

        # 第g世代
        self.X_g = self.X

    def assess(self):
        """評価"""
        H = []
        for xx in self.X_g:
            H.append(Simulator(xx).MaxAltitude())
        H = np.array(H)

        self.values.append(np.max(H))
        self.value_g = H
        self.val_sorted_index = np.argsort(H)[::-1]

    def select(self):
        """選択"""
        # エリート選択で4個，ルーレット選択で2個選ぶ
        # エリート選択
        elite_index = self.val_sorted_index[0:self.n_elite]
        not_elite_index = self.val_sorted_index[self.n_elite:]

        # ルーレット選択
        if np.argwhere(self.value_g[not_elite_index] > 0).size >= 2:
            weight = [self.value_g[nei] / np.sum(self.value_g[not_elite_index])
                      for nei in not_elite_index]
            roulette_index = \
                rand.choice(not_elite_index, self.n_roulette, replace=False,
                            p=weight)

        else:
            roulette_index = rand.choice(not_elite_index, self.n_roulette,
                                         replace=False)

        self.selected_index = np.r_[elite_index, roulette_index]

    def crossover2p(self):
        """交叉"""
        # 選択された6個の遺伝子を2点交叉により16個の遺伝子を作る
        ng = []
        for i in xrange(self.M - self.n_elite):
            # ランダムに2個の遺伝子を選ぶ
            parents_index = rand.choice(self.selected_index, 2, replace=False)
            # ランダムに交叉点を選ぶ
            cross_point = np.sort(
                rand.choice(np.arange(1, 3), 2, replace=False))
            # 2点交叉
            new_gene1 = self.X_g[parents_index[0]][0:cross_point[0]]
            new_gene2 = self.X_g[parents_index[1]][cross_point[0]:cross_point[1]]
            new_gene3 = self.X_g[parents_index[0]][cross_point[1]:]
            new_gene = np.r_[new_gene1, new_gene2, new_gene3]

            ng.append(new_gene)
        ng = np.array(ng)
        self.next_generation = np.r_[ng, self.X_g[self.selected_index[0:4]]]

    def mutation(self):
        """突然変異"""
        # 確率的に選ばれた染色体をランダム値に変更
        # 突然変異を起こす染色体の数
        n_mutation = rand.binomial(n=self.M * self.N, p=self.p_mutation)

        # 突然変異
        for i in xrange(n_mutation):
            m = rand.randint(self.M)
            n = rand.randint(self.N)
            if n == 0:
                self.next_generation[m][n] = rand.randint(5)
            elif n == 1:
                self.next_generation[m][n] = 0.05 + rand.rand() * 0.5
            elif n == 2:
                self.next_generation[m][n] = 0.7 + rand.rand() * 7
            elif n == 3:
                self.next_generation[m][n] = rand.rand()

    def alternation(self):
        """世代交代"""
        self.X_g = self.next_generation

    def best_gene(self):
        """最優秀遺伝子"""
        return self.X_g[self.val_sorted_index[0]]


if __name__ == '__main__':

    """設定"""
    # 計算世代
    G = 1000

    """計算"""
    ga = GeneticAlgorithm()
    for g in xrange(G):
        # 評価
        ga.assess()
        # 選択
        ga.select()
        # 交叉
        ga.crossover2p()
        # 突然変異
        ga.mutation()
        # 世代交代
        ga.alternation()

    # 最終世代
    ga.assess()
    best_gene = np.append(ga.best_gene(), ga.values[-1])
    np.savetxt("Best_Gene.csv", best_gene, fmt='%.8f', delimiter=',')

    # グラフ描画
    plt.plot(xrange(G + 1), ga.values)
    plt.title("Altitude of Rocket designed by Genetic Algorithm")
    plt.xlabel("Generation")
    plt.ylabel("Altitude[m]")
    plt.grid()
    plt.savefig("altitude_ga.png")
    # plt.show()
