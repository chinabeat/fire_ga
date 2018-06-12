# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import math
from math import factorial


#被攻击舰对目标的拦截次数
def N_b(V_m,V_w,t_p,t_f,R_max,R_min,n_d,s):
    n_b= 1/(math.log(1+V_m/V_w))*math.log((R_max+V_w*(t_p+t_f))/(R_min+V_w*(t_p+t_f)))+1
    n_b=int(n_b)
    return min(n_b,int(n_d/s))

#掩护舰船区域防空导弹对目标的拦截次数
def T_jg(t_jg,t_z,n):
    if n==1:
        return t_jg
    if n>1:
        return t_jg+t_z


def N_y(D,theta,V_m,V_w,phi_B,R_qmax,R_jmax,P_max,T_jg,n_d,s):
    K = 0
    P = D*math.sin(theta-phi_B)
    # print(P)
    X_B = P
    Y_B = D * math.cos(theta - phi_B)
    # print("{}:{}".format(X_B, Y_B))
    if P <= P_max:
        K = 1
        OT_2=R_qmax
        while True:
            X_T1 = P
            Y_T1 = math.sqrt(OT_2 ** 2 - P ** 2)
            # print("{}:{}".format(X_T1, Y_T1))

            X_F1=P
            Y_F1=Y_T1+R_qmax/V_w*V_m
            # print("{}:{}".format(X_F1, Y_F1))

            X_F2=P
            Y_F2=Y_F1-V_m*T_jg
            # print("{}:{}".format(X_F2, Y_F2))

            theta_f2=math.atan(X_F2/Y_F2)
            OF_2=math.sqrt(X_F2**2+Y_F2**2)
            BF_2=Y_F2-Y_B
            theta_tao2=math.asin(V_m/V_w*math.sin(theta_f2))
            # print(theta_f2,theta_tao2)
            OT_2=OF_2*math.sin(theta_f2)/math.sin(math.pi-theta_f2-theta_tao2)
            BT_2=math.sqrt(OT_2**2-P**2)-Y_B
            # print(OT_2)
            if BT_2 < R_jmax:
                break
            else:
                K+=1
            # print("K:{}".format(K))
    return min(K,int(n_d/s))

#命中弹丸数
def P_m(u,R,r_0):
    pass
#目标多个方向来袭
# T_hat_jg=T_jg+t_z
# N_m=P_m* n * t
# W=N_b+N_y+N_m

##############
#舰队防御模型说明
##############

def calc_arrangement(n, m):
    '''计算排列数'''
    return(factorial(n) / factorial(n - m))

def calc_combination(n, m):
    '''计算组合数'''
    return(calc_arrangement(n, m) / factorial(m))
def template_P(n,m,p):
    # print(calc_combination(n, m) ,pow(p, m) , pow((1 - p), (n - m)))
    return calc_combination(n, m) * pow(p, m) * pow((1 - p), (n - m))

#可靠飞行及目标捕获流程
def P_1i(N_0,i,P1):
    return template_P(N_0,i,P1)
#各类干扰流程
def P_2i(N_0,i,P2,P_1_list):
    result=0
    for j in range(i,N_0+1):
        result += template_P(j,i,P2)*P_1_list[j]
    return result
#导弹拦截

#经过一次防空导弹拦截后，N1枚导弹中有N2枚导弹成功突防的概率
def P_N1_N2(N_1,N_2,n,P3):
    if N_2<=N_1 and N_2>=0 and N_1<=n:
        return template_P(N_1,N_2,P3)
    if N_2<=N_1 and N_2>=N_1-n and N_1>n:
        return template_P(n, N_2-N_1+n, P3)
    if N_2<=N_1 and N_2<N_1-n and N_1>n:
        return 0

#k次拦截后，有i枚导弹成功突防的概率
def P_3_k(N_0,n,P_2_list,k_max,P3):
    P_3_k_list=P_2_list
    for k in range(k_max):
        P_3_k_subtract_1_list = P_3_k_list
        P_3_k_list = list()
        for i in range(N_0 + 1):
            P_3_i_k = 0
            for j in range(i, N_0 + 1):
                P_3_i_k += P_N1_N2(j, i, n, P3) * P_3_k_subtract_1_list[j]
            P_3_k_list.append(P_3_i_k)
        return P_3_k_list

    # P_3_k_list=list()
    # for i in range(N_0+1):
    #     P_3_i_k = 0
    #     for j in range(i,N_0+1):
    #         P_3_i_k += P_N1_N2(j, i, n, P3) * P_2_list[j]
    #     P_3_k_list.append(P_3_i_k)
    #
    # print(P_3_k_list)
    # if k_max==1 :
    #     return P_3_k_list
    # else:
    #     for k in range(2,k_max+1):
    #         P_3_k_subtract_1_list = P_3_k_list
    #         P_3_k_list = list()
    #         for i in range(N_0 + 1):
    #             P_3_i_k = 0
    #             for j in range(i, N_0 + 1):
    #                 P_3_i_k += P_N1_N2(j, i, n, P3) * P_3_k_subtract_1_list[j]
    #             P_3_k_list.append(P_3_i_k)
    #         return P_3_k_list
#密集阵拦截
def P_4_i(N_0,i,n,P_3_list,P4):
    result=0
    for j in range(i,N_0+1):
        result += P_N1_N2(j,i,n,P4)*P_3_list[j]
    return result
#弹群重创目标概率计算
def P(m,N_0,P_4_list):
    result=0
    for j in range(m,N_0+1):
        result += P_4_list[j]
    return result

if __name__=='__main__':
    V_m=100
    V_w=1000
    t_p=2
    t_f=2
    R_max=5000
    R_min=200
    n_d=30
    s=3
#被攻击舰对目标的拦截次数
    c_b=N_b(V_m,V_w,t_p,t_f,R_max,R_min,n_d,s)
    print('c_b:{}'.format(c_b))
# 掩护舰船区域防空导弹对目标的拦截次数
    target_nums=3
    t_z=2
    t_jg=2
    T_jg=T_jg(t_jg, t_z, target_nums)
    print("T_jg:{}".format(T_jg))
    D=2000
    theta=math.pi*4/5
    phi_B=math.pi*1/4
    R_qmax=5000
    R_jmax=1000
    P_max=3000
    c_y=N_y(D, theta, V_m, V_w, phi_B, R_qmax, R_jmax, P_max, T_jg, n_d, s)
    print("c_y:{}".format(c_y))

#舰队防御模型说明
    N_0=10
    P1=0.98
    P_1_list=list()
    for N_1 in range(N_0+1):
        P_1_list.append(P_1i(N_0,N_1,P1))
        # print('P_1i: {}'.format(P_1i(N_0,i,P1)))
    print("P_1_list:{}".format(P_1_list))

    P2=0.8
    P_2_list=list()
    for N_2 in range(N_0+1):
        P_2_list.append(P_2i(N_0,N_2,P2,P_1_list))
    print("P_2_list:{}".format(P_2_list))
    k_max=c_y+c_b
    P3 = 0.5
    # P_3_matrix=np.zeros((N_0,k_max))
    n= 8  # 火力通道数
    # N_4=3

    # print(P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3))
    P_3_list=P_3_k(N_0,n,P_2_list,k_max,P3)
    # for N_4 in range(N_0+1):
    #     P_3_list.append(P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3))
    print("P_3_list:{}".format(P_3_list))

    # for N_4 in range(N_0+1):
    #     for k in range(1,k_max+1):
    #         P_3_matrix[N_4][k]=P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3)
    # print(P_3_matrix)

    P4=0.5
    P_MJZ=0.7
    N_MJZ=10
    T_MJZ=5
    n_mijizhen=P_MJZ*N_MJZ*T_MJZ
    P_4_list=list()
    for i in range(N_0+1):
        P_4_list.append(P_4_i(N_0, i, n_mijizhen, P_3_list, P4))
    print("P_4_list:{}".format(P_4_list))

    m=2
    P_m=P(m,N_0,P_4_list)
    print("P_m:{}".format(P_m))


# N_i=N_b(V_m,V_w,t_p,t_f,R_max,R_min,n_d,s)
# for j in range(nums):
#     N_i += N_y(D,theta,V_m,V_w,phi_B,R_qmax,R_max[j],P_max,T_g[j],n_d,s)
# print(N_i)