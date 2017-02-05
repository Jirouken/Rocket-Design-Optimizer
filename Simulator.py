# -*- coding: utf-8 -*-

import numpy as np

class Simulator(object):

    def __init__(self, x):
        self.e = int(x[0])
        self.d = x[1]
        self.L = x[2]
        self.m_w = x[3]

    def MaxAltitude(self):
        e = self.e
        d = self.d
        L = self.L
        m_w = self.m_w

        # 設定
        g = 9.81        # 重力加速度[m/s^2]
        ad = 1.29       # 空気密度[kg/m^3]
        m_cansat = 0.2  # CanSat質量[kg]
        dt = 0.1        # 時間ステップ[s]
        n = 500         # 計算回数

        # 機体
        A = np.pi * (d / 2) ** 2    # 代表面積[m^2]
        rho = 0.8                   # 機体平均線密度[kg/m]
        m_airframe = rho * L
        CP = 0.75 * L               # 圧力中心
        Cd = 0.50                   # 抗力係数

        # エンジン(AiroTech G)
        G = np.genfromtxt("G_engine.txt",
                          dtype=("S3", float, float, float, float))
        tb = G[e][1]        # 燃焼時間[s]
        Ft_ave = G[e][2]    # 平均推力[N]
        W_total = G[e][3]   # 全質量[g]
        W_prop = G[e][4]    # 燃料質量[g]

        # 初期全質量
        ma = m_airframe + m_cansat + m_w + W_total / 1000
        # 初期重心
        CG = L * (0.7 * m_airframe + 0.3 * m_cansat + 0.1 * m_w + 0.95 * W_total / 1000) / ma
        # 初期安定性
        stb = (CP - CG) / d
        # 安定性効果
        se = 1 - (np.exp((1 - stb) * (1 - stb)) - 1) / (np.exp(1) - 1)

        # 推力
        def Ft(t):
            if t <= tb:
                return Ft_ave
            elif t > tb:  # burning finished
                return 0.0

        # エンジン質量[g]
        def mb(t):
            if t <= tb:
                return (W_total * 2 - W_prop) / 2  # average weight[g]
            elif t > tb:  # burning finished
                return W_total - W_prop

        # 全質量[kg]
        def m(t):
            return ma + mb(t) / 1000.

        # 抗力[N]
        def Fd(v):
            if v >= 0.0:
                return - Cd * ad * A * v * v / 2
            elif v < 0.0:
                return Cd * ad * A * v * v / 2

        # 運動方程式
        def a(t, v):
            return (Ft(t) + Fd(v)) / m(t) - g

        y0 = 0.0
        v0 = 0.0

        # Runge-Kutta method
        for i in range(n):
            t0 = dt * i
            ak1 = a(t0, v0)
            vk1 = v0
            ak2 = a(t0 + dt / 2, vk1 + dt / 2 * ak1)
            vk2 = vk1 + dt / 2 * ak1
            ak3 = a(t0 + dt / 2, vk2 + dt / 2 * ak2)
            vk3 = vk2 + dt / 2 * ak2
            ak4 = a(t0 + dt, vk3 + dt * ak3)
            vk4 = vk3 + dt * ak3
            v1 = v0 + dt * (ak1 + 2 * ak2 + 2 * ak3 + ak4) / 6
            y1 = y0 + dt * (vk1 + 2 * vk2 + 2 * vk3 + vk4) / 6
            v0 = v1
            y0 = y1
            if v1 < 0.0:
                break

        if y0 < 0:
            y0 = 0.

        res = y0 * se

        if res < 0:
            res = 0.

        return res
