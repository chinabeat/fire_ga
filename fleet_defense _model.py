# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import math
from math import factorial

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
    k_max=10
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