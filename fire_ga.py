# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import math
from math import factorial

# The number of intercepts by the attacked ship against the target
# V_m  The average speed of the incoming target flight
# V_w  Average speed of regional air defense missiles
# t_p  Intercept effectiveness evaluation time
# t_f  Regional air defense missile launch preparation time
# R_max  Regional air defense missile deterrence zone
# R_min  Regional Bound Zone of Air Defense Missiles
# n_d  Resilience
# s  The number of salvo missiles
def N_b(V_m,V_w,t_p,t_f,R_max,R_min,n_d,s):
    n_b= 1/(math.log(1+V_m/V_w))*math.log((R_max+V_w*(t_p+t_f))/(R_min+V_w*(t_p+t_f)))+1
    n_b=int(n_b)
    return min(n_b,int(n_d/s))

# Air defense missile launch interval
# Total air-to-air missile launch interval
#t_jg Air Defense Missile Launch Interval
# t_z transfer time
def T_jg_hat(t_jg,t_z,n):
    if n==1:
        return t_jg
    if n>1:
        return t_jg+t_z

# The number of intercepts a cover ship makes against a target
# D  Queue spacing
# theta  Formation Queue Angle
# phi_B  incoming target relative to the attack ship’s entry angle
# V_m  Strike target flying speed
# V_w  Vessel Ship-to-Air Missile Flight Speed
# R_qmax  Demarcation of the target area of the ship-to-air missile of the shield ship
# R_jmax  Breach of the Kill Zone of the Short-range Air Defense Weapon System of the Assaulted Ship
# P_max  The maximum route shortcut for the ship-to-air missile to the target
# T_jg_hat  Total Launch Interval of Air Defense Missiles
# n_d  The amount of charge
# s  number of salvos per time

def N_y(D,theta,V_m,V_w,phi_B,R_qmax,R_jmax,P_max,T_jg_hat,n_d,s):
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
            Y_F2=Y_F1-V_m*T_jg_hat
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

#Number of hit bullets
def P_m(u,R,r_0):
    pass


##############
#Fleet defense model description
##############

def calc_arrangement(n, m):
    return(factorial(n) / factorial(n - m))

def calc_combination(n, m):
    return(calc_arrangement(n, m) / factorial(m))

def template_P(n,m,p):
    # print(calc_combination(n, m) ,pow(p, m) , pow((1 - p), (n - m)))
    return calc_combination(n, m) * pow(p, m) * pow((1 - p), (n - m))
# Global parameters
# N_0 missile initial charge
# i Number of Missiles Passed


# Reliable flight and target capture process
# P1  The probability that a single missile can reliably fly and successfully capture a target
# N_0 missile initial charge
# i Number of Missiles Passed
def P_1i(N_0,i,P1):
    return template_P(N_0,i,P1)
#Various types of interference process
# P2 Probability of Successful Single-Mile Missile Interference
# P_1_list  ProSuccessful Missilebability Table for the First Phase
# N_0 missile initial charge
# i Number of Missiles Passed
def P_2i(N_0,i,P2,P_1_list):
    result=0
    for j in range(i,N_0+1):
        result += template_P(j,i,P2)*P_1_list[j]
    return result
# Missile interception

# After an air defense missile interception, the probability of successful penetration of N2 missiles in N1 missiles
# P3  The probability of a single missile breaking through the air defense missile interception
# n  Available fire access for this interception
def P_N1_N2(N_1,N_2,n,P3):
    if N_2<=N_1 and N_2>=0 and N_1<=n:
        return template_P(N_1,N_2,P3)
    if N_2<=N_1 and N_2>=N_1-n and N_1>n:
        return template_P(n, N_2-N_1+n, P3)
    if N_2<=N_1 and N_2<N_1-n and N_1>n:
        return 0

# After k interceptions, the probability of successful missile penetration by i missiles
# N_0  missile initial charge
# P3  The probability of a single missile breaking through the air defense missile interception
# n  Available fire access for this interception
# P_2_list  ProSuccessful Missilebability Table for the Second Phase
# k_max  The maximum number of interceptions for air defense missiles
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

#Dense array interception
# n Dense array of multi-target processing capabilities
# P4 Probability of Successful Penetration When Faced with a Dense Block
# P_3_list Probability Table for Missiles Passed in Phase III
def P_4_i(N_0,i,n,P_3_list,P4):
    result=0
    for j in range(i,N_0+1):
        result += P_N1_N2(j,i,n,P4)*P_3_list[j]
    return result
#The elastic group re-creates the target probability calculation
# m The number of successful penetration missiles that can hit the target
# P_4_list Probability Table for Missiles Passed in Phase IV
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
#The number of times the attacked ship intercepted the target
    c_b=N_b(V_m,V_w,t_p,t_f,R_max,R_min,n_d,s)
    print('c_b:{}'.format(c_b))
# The number of times of interception of targets by the warship’s regional air defense missiles
    target_nums=3
    t_z=2
    t_jg=2
    T_jg_hat=T_jg_hat(t_jg, t_z, target_nums)
    print("T_jg:{}".format(T_jg_hat))
    D=2000
    theta=math.pi*4/5
    phi_B=math.pi*1/4
    R_qmax=10000
    R_jmax=5000
    P_max=3000
    c_y=N_y(D, theta, V_m, V_w, phi_B, R_qmax, R_jmax, P_max, T_jg_hat, n_d, s)
    print("c_y:{}".format(c_y))

#Fleet defense model description
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
    n= 8  # Fire channel number
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