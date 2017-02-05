# -*- coding: utf-8 -*-

import numpy as np
from numpy import random as rand
from Simulator import Simulator
from matplotlib import pyplot as plt

"""設定"""
# 染色体数
N = 4
# 個体数
M = 20
# エリート選択個数
n_elite = 4
# ルーレット選択個数
n_roulette = 2
# 突然変異を起こす確率
p_mutation = 0.05
# 計算世代
G = 1000

"""初期値"""
X = []
for i in xrange(M):
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
    X.append(x)
X = np.array(X)


"""遺伝的アルゴリズム"""
# 第g世代
X_g = X
# 第g世代評価
value_g = []
# 第g世代評価降順インデックス
val_sorted_index = []
# 最大評価値
values = []
# 選択された遺伝子のインデックス
selected_index = []
# 次世代遺伝子
next_generation = []


def assess():
    """評価"""
    global value_g, val_sorted_index

    H = []
    for xx in X_g:
        H.append(Simulator(xx).MaxAltitude())
    H = np.array(H)

    values.append(np.max(H))
    value_g = H
    val_sorted_index = np.argsort(H)[::-1]


def select():
    """選択"""
    # エリート選択で4個，ルーレット選択で2個選ぶ
    global selected_index

    # エリート選択
    elite_index = val_sorted_index[0:n_elite]
    not_elite_index = val_sorted_index[n_elite:]

    # ルーレット選択
    if np.argwhere(value_g[not_elite_index] > 0).size >= 2:
        weight = [value_g[nei] / np.sum(value_g[not_elite_index])
                  for nei in not_elite_index]
        roulette_index = \
            rand.choice(not_elite_index, n_roulette, replace=False,
                        p=weight)

    else:
        roulette_index = rand.choice(not_elite_index, n_roulette,
                                     replace=False)

    selected_index = np.r_[elite_index, roulette_index]


def crossover2p():
    """交叉"""
    # 選択された6個の遺伝子を2点交叉により16個の遺伝子を作る
    global next_generation

    ng = []
    for i in xrange(M - n_elite):
        # ランダムに2個の遺伝子を選ぶ
        parents_index = rand.choice(selected_index, 2, replace=False)
        # ランダムに交叉点を選ぶ
        cross_point = np.sort(
            rand.choice(np.arange(1, 3), 2, replace=False))
        # 2点交叉
        new_gene1 = X_g[parents_index[0]][0:cross_point[0]]
        new_gene2 = X_g[parents_index[1]][cross_point[0]:cross_point[1]]
        new_gene3 = X_g[parents_index[0]][cross_point[1]:]
        new_gene = np.r_[new_gene1, new_gene2, new_gene3]

        ng.append(new_gene)
    ng = np.array(ng)
    next_generation = np.r_[ng, X_g[selected_index[0:4]]]


def mutation():
    """突然変異"""
    # 確率的に選ばれた染色体をランダム値に変更
    global next_generation

    # 突然変異を起こす染色体の数
    n_mutation = rand.binomial(n=M * N, p=p_mutation)

    # 突然変異
    for i in xrange(n_mutation):
        m = rand.randint(M)
        n = rand.randint(N)
        if n == 0:
            next_generation[m][n] = rand.randint(5)
        elif n == 1:
            next_generation[m][n] = 0.05 + rand.rand() * 0.5
        elif n == 2:
            next_generation[m][n] = 0.7 + rand.rand() * 7
        elif n == 3:
            next_generation[m][n] = rand.rand()


for g in xrange(G):
    # 評価
    assess()
    # 選択
    select()
    # 交叉
    crossover2p()
    # 突然変異
    mutation()
    # 世代交代
    X_g = next_generation

# 最終世代評価
assess()
best_gene = X_g[val_sorted_index[0]]

best_gene = np.append(best_gene, values[-1])
np.savetxt("Best_Gene.csv", best_gene, fmt='%.8f', delimiter=',')

# グラフ描画
plt.plot(xrange(G+1), values)
plt.title("Altitude of Rocket designed by Genetic Algorithm")
plt.xlabel("Generation")
plt.ylabel("Altitude[m]")
plt.grid()
plt.savefig("altitude_ga.png")
# plt.show()
