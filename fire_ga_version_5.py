# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import math
from math import factorial


##############
# Fleet defense model description
##############

def calc_arrangement(n, m):
    return (factorial(n) / factorial(n - m))


def calc_combination(n, m):
    return (calc_arrangement(n, m) / factorial(m))


def template_P(n, m, p):
    # print(calc_combination(n, m) ,pow(p, m) , pow((1 - p), (n - m)))
    return calc_combination(n, m) * pow(p, m) * pow((1 - p), (n - m))


# Global parameters
# N_0 missile initial charge
# i Number of Missiles Passed


# Reliable flight and target capture process
# P1  The probability that a single missile can reliably fly and successfully capture a target
# N_0 missile initial charge
# i Number of Missiles Passed
def P_1i(N_0, i, P1):
    return template_P(N_0, i, P1)


# Various types of interference process
# P2 Probability of Successful Single-Mile Missile Interference
# P_1_list  ProSuccessful Missilebability Table for the First Phase
# N_0 missile initial charge
# i Number of Missiles Passed
def P_2i(N_0, i, P2, P_1_list):
    result = 0
    for j in range(i, N_0 + 1):
        result += template_P(j, i, P2) * P_1_list[j]
    return result


# Missile interception

# After an air defense missile interception, the probability of successful penetration of N2 missiles in N1 missiles
# P3  The probability of a single missile breaking through the air defense missile interception
# n  Available fire access for this interception
def P_N1_N2(N_1, N_2, n, P3):
    if N_2 <= N_1 and N_2 >= 0 and N_1 <= n:
        return template_P(N_1, N_2, P3)
    if N_2 <= N_1 and N_2 >= N_1 - n and N_1 > n:
        return template_P(n, N_2 - N_1 + n, P3)
    if N_2 <= N_1 and N_2 < N_1 - n and N_1 > n:
        return 0


# After k interceptions, the probability of successful missile penetration by i missiles
# N_0  missile initial charge
# P3  The probability of a single missile breaking through the air defense missile interception
# n  Available fire access for this interception
# P_2_list  ProSuccessful Missilebability Table for the Second Phase
# k_max  The maximum number of interceptions for air defense missiles
def P_3_k(N_0, n, P_3_input_list, P3):
    # P_3_k_list=P_2_list
    P_3_k_subtract_1_list = P_3_input_list
    P_3_k_list = list()
    for i in range(N_0 + 1):
        P_3_i_k = 0
        for j in range(i, N_0 + 1):
            P_3_i_k += P_N1_N2(j, i, n, P3) * P_3_k_subtract_1_list[j]
        P_3_k_list.append(P_3_i_k)
    return P_3_k_list


# Dense array interception
# n Dense array of multi-target processing capabilities
# P4 Probability of Successful Penetration When Faced with a Dense Block
# P_3_list Probability Table for Missiles Passed in Phase III
def P_4_i(N_0, i, n, P_3_list, P4):
    result = 0
    for j in range(i, N_0 + 1):
        result += P_N1_N2(j, i, n, P4) * P_3_list[j]
    return result


# The elastic group re-creates the target probability calculation
# m The number of successful penetration missiles that can hit the target
# P_4_list Probability Table for Missiles Passed in Phase IV
def P(m, N_0, P_4_list):
    result = 0
    for j in range(m, N_0 + 1):
        result += P_4_list[j]
    return result


class fleet(object):
    # fire_num:Number of fire channels
    # distance:Intercept distance , km unit
    # time:Fire channel occupancy time , s unit
    # x:x Axis coordinate , km unit
    # y:y axis coordinate , km unit
    def __init__(self, fire_num, distance, oc_time, x, y, dead=False):
        self.fire_num = fire_num
        self.distance = distance
        self.oc_time = oc_time
        self.freeze_time=0
        self.x = x
        self.y = y
        self.dead = dead
        # self.res_fire_num=fire_num

    # the distance between the fleet and missile
    # x : x axis coordinate of missile
    # y : y axis coordinate of missile
    def distance_to_missile(self, missile):
        return math.sqrt((missile.x - self.x) ** 2 + (missile.y - self.y) ** 2)

    def freeze(self):
        self.freeze_time=self.oc_time
    def unfreezing(self):
        self.freeze_time -= 1

    def able_fire(self):


        pass

    def occupy_fire_channel(self):
        self.fire_num -= 1

    def relieve_fire_channel(self):
        self.fire_num += 1

    def died(self):
        self.dead = True


class missile(object):
    def __init__(self, x, y, target, velocity=0.2):
        self.x = x
        self.y = y
        # self.theta=theta
        self.velocity = velocity
        self.target = target
        self.intercepted_num = 0

    def get_theta(self, fleet):
        if self.x - fleet.x == 0:
            if self.y - fleet.y>0:
                return math.pi*0.5
            elif self.y - fleet.y==0:
                return 0.0
            else:
                return -0.5*math.pi
        else:
            return math.atan((self.y - fleet.y) / (self.x - fleet.x))

    def update_coordinate(self, time, fleet):
        if self.x >= fleet.x and self.y >= fleet.y:
            theta = self.get_theta(fleet)
            self.x = self.x - self.velocity * time * math.cos(theta)
            self.y = self.y - self.velocity * time * math.sin(theta)
        elif self.x < fleet.x and self.y >= fleet.y:
            theta = math.atan2(self.y - fleet.y, fleet.x - self.x)
            self.x = self.x + self.velocity * time * math.cos(theta)
            self.y = self.y - self.velocity * time * math.sin(theta)
        elif self.x < fleet.x and self.y < fleet.y:
            theta = math.atan2(fleet.y - self.y, fleet.x - self.x)
            self.x = self.x + self.velocity * time * math.cos(theta)
            self.y = self.y + self.velocity * time * math.sin(theta)
        else:
            theta = math.atan2(fleet.y - self.y, self.x - fleet.x)
            self.x = self.x - self.velocity * time * math.cos(theta)
            self.y = self.y + self.velocity * time * math.sin(theta)

    def intercepted(self):
        self.intercepted_num += 1

    def reset_intercepted_num(self):
        self.intercepted_num = 0


def simulator(num_missile):
    pass


def step(action):
    pass



# 一枚导弹攻击A舰船

# Fleet defense model description
#     N_0=1
#     P1=0.98
#     P_1_list=list()
#     for N_1 in range(N_0+1):
#         P_1_list.append(P_1i(N_0,N_1,P1))
#         # print('P_1i: {}'.format(P_1i(N_0,i,P1)))
#     print("P_1_list:{}".format(P_1_list))
#
#     P2=0.8
#     P_2_list=list()
#     for N_2 in range(N_0+1):
#         P_2_list.append(P_2i(N_0,N_2,P2,P_1_list))
#     print("P_2_list:{}".format(P_2_list))
#     k_max=10
#     P3 = 0.5
#     # P_3_matrix=np.zeros((N_0,k_max))
#     n= 8  # Fire channel number
#     # N_4=3
#
#     # print(P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3))
#     P_3_list=P_3_k(N_0,n,P_2_list,k_max,P3)
#     # for N_4 in range(N_0+1):
#     #     P_3_list.append(P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3))
#     print("P_3_list:{}".format(P_3_list))
#
#     # for N_4 in range(N_0+1):
#     #     for k in range(1,k_max+1):
#     #         P_3_matrix[N_4][k]=P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3)
#     # print(P_3_matrix)
#
#     P4=0.5
#     P_MJZ=0.7
#     N_MJZ=10
#     T_MJZ=5
#     n_mijizhen=P_MJZ*N_MJZ*T_MJZ
#     P_4_list=list()
#     for i in range(N_0+1):
#         P_4_list.append(P_4_i(N_0, i, n_mijizhen, P_3_list, P4))
#     print("P_4_list:{}".format(P_4_list))
#
#     m=2
#     P_m=P(m,N_0,P_4_list)
#     print("P_m:{}".format(P_m))

# N_i=N_b(V_m,V_w,t_p,t_f,R_max,R_min,n_d,s)
# for j in range(nums):
#     N_i += N_y(D,theta,V_m,V_w,phi_B,R_qmax,R_max[j],P_max,T_g[j],n_d,s)
# print(N_i)

class env():
    def __init__(self):
        self.fleet_a = fleet(8, 20, 20, 18, 8 + 6 * math.sqrt(3))
        self.fleet_b = fleet(8, 20, 20, 24, 8 + 4 * math.sqrt(3))
        self.fleet_c = fleet(4, 15, 25, 12, 8 + 4 * math.sqrt(3))
        self.fleet_d = fleet(4, 15, 25, 30, 8 + 2 * math.sqrt(3))
        self.fleet_e = fleet(4, 15, 25, 6, 8 + 2 * math.sqrt(3))
        self.fleet_f = fleet(4, 15, 25, 36, 8)
        self.fleet_g = fleet(4, 15, 25, 0, 8)
        self.fleet_h = fleet(4, 15, 25, 26, 0)
        self.fleet_i = fleet(4, 15, 25, 10, 0)
        self.fleet_j = fleet(4, 15, 25, 18, 8)
        self.fleet_dict = {
            0: self.fleet_a,
            1: self.fleet_b,
            2: self.fleet_c,
            3: self.fleet_d,
            4: self.fleet_e,
            5: self.fleet_f,
            6: self.fleet_g,
            7: self.fleet_h,
            8: self.fleet_i,
            9: self.fleet_j
        }
        # self.fleet_dict = {
        #               "a": self.fleet_a,
        #               "b": self.fleet_b,
        #               "c": self.fleet_c,
        #               "d": self.fleet_d,
        #               "e": self.fleet_e,
        #               "f": self.fleet_f,
        #               "g": self.fleet_g,
        #               "h": self.fleet_h,
        #               "i": self.fleet_i,
        #               "j": self.fleet_j
        #               }
        self.state_dim = len(self.fleet_dict)
        self.action_dim = len(self.fleet_dict)*20
        self.state = [1]*self.state_dim
        self.done = False
        self.done_state = [0 for i in range(len(self.fleet_dict))]
    def reset(self):
        self.state = [1]*self.state_dim
        self.done = False
        for v in self.fleet_dict.values():
            v.dead = False
        return [1]*self.state_dim
    def step(self,action):
        # a to a
        fleet_dict = self.fleet_dict
        # action 10 dims
        # coordinates = list()
        coordinates = [[x, y] for x in range(-20, 61, 20) for y in range(-20, 41, 20)]
        coordinates_count = len(coordinates)
        targets = [i for i in range(len(fleet_dict))]
        # targets_count = len(targets)

        # action = action
        coordinate = coordinates[action % coordinates_count]
        target = targets[int(action / coordinates_count)]
        print (coordinate, target)
        missile_a = missile(coordinate[0], coordinate[1], target, 0.2)
        if fleet_dict[target].dead == True:
            P_m = -1000.0
            # state = self.state
            reward = P_m
            # done = self.done
            return self.state,reward,self.done

        # fleet_list = [fleet_a, fleet_b, fleet_c, fleet_d, fleet_e, fleet_f, fleet_g, fleet_h, fleet_i, fleet_j]
        # fleet_list_temp = fleet_list


        N_0 = 1
        P1 = 0.98
        P_1_list = list()
        for N_1 in range(N_0 + 1):
            P_1_list.append(P_1i(N_0, N_1, P1))
            # print('P_1i: {}'.format(P_1i(N_0,i,P1)))
        print("P_1_list:{}".format(P_1_list))

        P2 = 0.8
        P_2_list = list()
        for N_2 in range(N_0 + 1):
            P_2_list.append(P_2i(N_0, N_2, P2, P_1_list))
        print("P_2_list:{}".format(P_2_list))
        P_3_input_list = P_2_list
        P_3_list = list()
        # missile_list = [missile_a]
        # theta_a_target=missile_a.get_theta(fleet_dict[missile_a.target])

        distance_a_target = fleet_dict[missile_a.target].distance_to_missile(missile_a)
        print ("distance_a_target:{}".format(distance_a_target))
        fly_time = 0
        for distance in np.arange(distance_a_target, 0, -1 * missile_a.velocity):
        # while distance_a_target > 0:
            for fleet in fleet_dict.keys():
                if fleet_dict[fleet].freeze_time > 0:
                    fleet_dict[fleet].unfreezing()
            if distance>fleet_dict[missile_a.target].distance or fleet_dict[missile_a.target].freeze_time>0:
                missile_a.update_coordinate(1,fleet_dict[missile_a.target])
                # distance_a_target = fleet_dict[missile_a.target].distance_to_missile(missile_a)
                continue
            n = 0
            fly_time += 1
            for fleet in fleet_dict.keys():
                if (fleet_dict[fleet].dead==False) and (fleet_dict[fleet].distance_to_missile(missile_a) < fleet_dict[fleet].distance and fleet_dict[fleet].freeze_time==0):
                    n += fleet_dict[fleet].fire_num
                    fleet_dict[fleet].freeze()
            print ("n:", n)


        # distance_a_target = fleet_dict[missile_a.target].distance_to_missile(missile_a)
        # print ("distance_a_target:{}".format(distance_a_target))
        # for distance in np.arange(distance_a_target, 0, -1 * missile_a.velocity * fleet_dict[missile_a.target].oc_time):
        #     n = 0  # Fire channel number
        #     # print ("distance:{}".format(distance))
        #     # print (missile_a.x, " ", missile_a.y)
        #     if distance > fleet_dict[missile_a.target].distance:
        #         missile_a.update_coordinate(fleet_dict[missile_a.target].oc_time, fleet_dict[missile_a.target])
        #         # print missile_a.x," ",missile_a.y
        #         continue
        #     for fleet in fleet_dict.keys():
        #         # print ("fleet {} to the missile distance is {} .".format(fleet,fleet_dict[fleet].distance_to_missile(missile_a)))
        #         if (fleet_dict[fleet].dead==False) and (fleet_dict[fleet].distance_to_missile(missile_a) < fleet_dict[fleet].distance ):
        #             n += fleet_dict[fleet].fire_num

            # print ("n:", n)
            P3 = 0.5
            # P_3_matrix=np.zeros((N_0,k_max))
            # n = 0  # Fire channel number

            # N_4=3

            # print(P_3_i_k(N_0,N_4,n,P_2_list,k_max,P3))
            P_3_list = P_3_k(N_0, n, P_3_input_list, P3)
            P_3_input_list = P_3_list
            missile_a.update_coordinate(fleet_dict[missile_a.target].oc_time, fleet_dict[missile_a.target])
            # print("P_3_list:{}".format(P_3_list))

        P4 = 0.5
        P_MJZ = 0.7
        N_MJZ = 10
        T_MJZ = 5
        n_mijizhen = P_MJZ * N_MJZ * T_MJZ
        P_4_list = list()
        for i in range(N_0 + 1):
            P_4_list.append(P_4_i(N_0, i, n_mijizhen, P_3_list, P4))
        print("P_4_list:{}".format(P_4_list))

        m = 1
        P_m = P(m, N_0, P_4_list)
        print("P_m:{}".format(P_m))
        if P_m > 0.3:
            fleet_dict[target].died()
            print ("Is fleet {} died? Answer:{} .".format(target, fleet_dict[target].dead))

            self.state[target]=0
            # next_state = self.state
            reward = P_m*100
            # done_state = [0 for i in range(len(fleet_dict))]
            if self.state == self.done_state:
                self.done = True
            # info =
            return self.state, reward, self.done
        else:
            reward = -10.0
            return self.state, reward, self.done

if __name__=="__main__":
    env=env()
    print (env.step(37))

    print (env.state)
    for fleet in env.fleet_dict.values():
        print (fleet.x," ",fleet.y)
    # print(env.fleet_dict)

    # fleet_a = fleet(8, 20, 20, 18, 8 + 6 * math.sqrt(3))
    # fleet_b = fleet(8, 20, 20, 24, 8 + 4 * math.sqrt(3))
    # fleet_c = fleet(4, 15, 25, 12, 8 + 4 * math.sqrt(3))
    # fleet_d = fleet(4, 15, 25, 30, 8 + 2 * math.sqrt(3))
    # fleet_e = fleet(4, 15, 25, 6, 8 + 2 * math.sqrt(3))
    # fleet_f = fleet(4, 15, 25, 36, 8)
    # fleet_g = fleet(4, 15, 25, 0, 8)
    # fleet_h = fleet(4, 15, 25, 26, 0)
    # fleet_i = fleet(4, 15, 25, 10, 0)
    # fleet_j = fleet(4, 15, 25, 18, 8)
    #
    # fleet_dict = {"a": fleet_a,
    #               "b": fleet_b,
    #               "c": fleet_c,
    #               "d": fleet_d,
    #               "e": fleet_e,
    #               "f": fleet_f,
    #               "g": fleet_g,
    #               "h": fleet_h,
    #               "i": fleet_i,
    #               "j": fleet_j
    #               }
    # state_initial = [0]*len(fleet_dict)



# missile_a = missile(-20,-20, "a", 0.2)
# missile_b = missile(100, 100, "a", 0.2)
# missile_c = missile(100, 100, "a", 0.2)
# missile_d = missile(100, 100, "a", 0.2)



