# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import os
import math
from math import factorial
import random
from itertools import combinations


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
    # print("P_3_list:{}".format(P_3_list))
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
        self.fire_channels=[0]*fire_num
        # self.res_fire_num=fire_num

    # the distance between the fleet and missile
    # x : x axis coordinate of missile
    # y : y axis coordinate of missile
    def distance_to_missile(self, missile):
        return math.sqrt((missile.x - self.x) ** 2 + (missile.y - self.y) ** 2)

    def reset_fire_channels(self):
        # print("reset_fire_channels ",self.fire_channels)
        self.fire_channels = [0] * self.fire_num

    def channel_unfreezing(self,channel):
        # print("channel_unfreezing ",self.fire_channels)
        self.fire_channels[channel] -=1

    def channel_can_fire_count(self):
        # print("chanel_can_fire_count ",self.fire_channels)
        return len(self.channel_can_fire())

    def channel_can_fire(self):
        channel_can_fire_number=list()
        for i in range(len(self.fire_channels)):
            if self.fire_channels[i] ==0:
                channel_can_fire_number.append(i)
        return channel_can_fire_number

    def use_channel(self,N_0):
        channel_can_fire_number=self.channel_can_fire()
        random.shuffle(channel_can_fire_number)
        if N_0<=len(channel_can_fire_number):
            for i in range(N_0):
                self.fire_channels[channel_can_fire_number[i]]=self.oc_time

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
    def __init__(self, x, y, target, numbers=1,velocity=0.2):
        self.x = x
        self.y = y
        # self.theta=theta
        self.velocity = velocity
        self.target = target
        self.numbers = numbers
        self.intercepted_num = 0
        self.P_1_list = list()
        self.P_2_list = list()
        self.P_3_input_list = list()
        self.P_3_list = list()
        self.P_4_list = list()
        self.P_m = 0
        self.state = [0]*59
        self.action = [0]*960
        self.next_state = [0]*59
        self.reward = -numbers*0.1
        self.done = False
        self.update_next_state = True
        self.step = 0
        self.episode = 0

    def print_all_member(self):
        for name,value in vars(self).items():
            print("{} = {}".format(name,value))

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
        self.fleet_j = fleet(0, 0, 0, 18, 8)
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

        self.flying_missiles_dict=dict()
        self.one_missile_fly_max_time = 500
        self.total_missiles = 100
        self.coordinates = [[x, -20] for x in range(-20, 61, 20)] + [[x, 60] for x in range(-20, 61, 20)] + [[-20, y] for y in range(0,41,20)] + [[60, y] for y in range(0, 41, 20)]
        # self.coordinates = [[x,-20] for x in range(-20, 61, 20)]+[[x,60] for x in range(-20,61,20)]+[[-20,y] for y in range(0,41,20)]+[[60,y] for y in range(0,41,20)]+[[]]
        # self.coordinates = [[-20,-20]]
        # print(len(self.coordinates))
        self.missile_number = range(1,6)
        # self.fire_coordinates = list()
        # temp = list()
        # for number in self.missile_number:
        #     temp += list(combinations(self.coordinates, number))
        # for t in temp:
        #     self.fire_coordinates.append(list(t))
        # print("len fire_coordinates:{}".format(len(self.fire_coordinates)))
        # print([] in self.fire_coordinates)
        # print(self.fire_coordinates)
        self.state = [1]*len(self.fleet_dict)+[time for f in self.fleet_dict.values() for time in f.fire_channels]+[self.total_missiles]
        self.state_dim = len(self.state)
        self.action_dim = len(self.fleet_dict)*len(self.coordinates)*len(self.missile_number)

        self.done = False
        self.done_state = [0 for i in range(len(self.fleet_dict))]

        self.P_total = [[0,0]]*len(self.fleet_dict)

        self.N_0 = 0

    def reset(self):
        print("reset state:")
        for fleet in self.fleet_dict.keys():
            self.fleet_dict[fleet].reset_fire_channels()
        self.total_missiles = 100
        self.state = [1]*len(self.fleet_dict)+[time for fleet in self.fleet_dict.values() for time in fleet.fire_channels]+[self.total_missiles]
        print("reset state:{}".format(self.state))
        self.done = False
        for v in self.fleet_dict.values():
            v.dead = False
        return self.state

    def step(self,action,step,episode):
        # a to a
        reward = 0
        replayer_list = list()
        will_be_pop_missile_list = list()
        # print("state",self.state)
        fleet_dict = self.fleet_dict
        # coordinates = [[x, y] for x in range(-20, 61, 20) for y in range(-20, 41, 20)]
        coordinates = self.coordinates
        coordinates_count = len(coordinates)
        targets = [i for i in range(len(fleet_dict))]
        # missile_number=range(1,6)
        missile_number = self.missile_number
        targets_count = len(targets)

        if action == len(coordinates) * len(fleet_dict)*len(missile_number):
            coordinate = coordinates[0]
            target = targets[0]
            N_0 = 0
        else:
            coordinate = coordinates[action % (coordinates_count * targets_count) % coordinates_count]
            target = targets[int(action % (coordinates_count * targets_count) / coordinates_count)]
            N_0 = missile_number[int(action / (coordinates_count * targets_count))]

        print ("coordinate:{}, target:{}, N_0:{}".format(coordinate, target,N_0))

        for fleet in fleet_dict.keys():
            for channel in range(len(fleet_dict[fleet].fire_channels)):
                if fleet_dict[fleet].fire_channels[channel] > 0:
                    fleet_dict[fleet].channel_unfreezing(channel)

        if N_0 == 0:
            missile_a = missile(coordinate[0], coordinate[1], target, N_0, 0.2)
            missile_a.state = self.state
            missile_a.action = action
            missile_a.step = step
            missile_a.episode = episode
            # missile_a.reward = -10
        else:
        # if self.total_missiles<N_0:
            missile_a = missile(coordinate[0], coordinate[1], target,N_0, 0.2)
            missile_a.state = self.state
            missile_a.action = action
            missile_a.step = step
            missile_a.episode = episode

            self.flying_missiles_dict[step] = missile_a
            self.total_missiles -= self.flying_missiles_dict[step].numbers

            P1 = 0.98
            P_1_list = list()
            for N_1 in range(self.flying_missiles_dict[step].numbers + 1):
                P_1_list.append(P_1i(self.flying_missiles_dict[step].numbers, N_1, P1))
                # print('P_1i: {}'.format(P_1i(N_0,i,P1)))
            self.flying_missiles_dict[step].P_1_list = P_1_list
            # print("P_1_list:{}".format(P_1_list))

            P2 = 0.8
            P_2_list = list()
            for N_2 in range(self.flying_missiles_dict[step].numbers + 1):
                P_2_list.append(P_2i(self.flying_missiles_dict[step].numbers, N_2, P2, P_1_list))
            self.flying_missiles_dict[step].P_2_list = P_2_list
            # print("P_2_list:{}".format(P_2_list))

            self.flying_missiles_dict[step].P_3_input_list = P_2_list




        for k in self.flying_missiles_dict.keys():
            if fleet_dict[self.flying_missiles_dict[k].target].distance_to_missile(self.flying_missiles_dict[k]) > 0.2:
                n = 0
                # fly_time += 1
                for fleet in fleet_dict.keys():
                    if (fleet_dict[fleet].dead == False) and (fleet_dict[fleet].distance_to_missile(self.flying_missiles_dict[k]) < fleet_dict[fleet].distance):
                        n += fleet_dict[fleet].channel_can_fire_count()
                if (fleet_dict[self.flying_missiles_dict[k].target].dead == False) and (fleet_dict[self.flying_missiles_dict[k].target].distance_to_missile(self.flying_missiles_dict[k]) <fleet_dict[self.flying_missiles_dict[k].target].distance):
                    if fleet_dict[self.flying_missiles_dict[k].target].channel_can_fire_count() >= self.flying_missiles_dict[k].numbers:
                        fleet_dict[self.flying_missiles_dict[k].target].use_channel(self.flying_missiles_dict[k].numbers)
                    else:
                        N_0_residual = self.flying_missiles_dict[k].numbers - fleet_dict[self.flying_missiles_dict[k].target].channel_can_fire_count()
                        fleet_dict[self.flying_missiles_dict[k].target].use_channel(fleet_dict[self.flying_missiles_dict[k].target].channel_can_fire_count())
                        fleets = [x for x in range(10)]
                        fleets.remove(target)
                        random.shuffle(fleets)
                        for fleet in fleets:
                            if fleet_dict[fleet].dead == False and fleet_dict[fleet].distance_to_missile(
                                    self.flying_missiles_dict[k]) < fleet_dict[fleet].distance:
                                if fleet_dict[fleet].channel_can_fire_count() >= N_0_residual:
                                    fleet_dict[fleet].use_channel(N_0_residual)
                                    break
                                else:
                                    N_0_residual -= fleet_dict[fleet].channel_can_fire_count()
                                    fleet_dict[fleet].use_channel(fleet_dict[fleet].channel_can_fire_count())

                else:
                    # fleet_dict[self.flying_missiles_dict[k].target].use_channel(fleet_dict[self.flying_missiles_dict[k].target].channel_can_fire_count())
                    N_0_residual = self.flying_missiles_dict[k].numbers
                    fleets = [x for x in range(10)]
                    fleets.remove(target)
                    random.shuffle(fleets)
                    for fleet in fleets:
                        if fleet_dict[fleet].dead == False and fleet_dict[fleet].distance_to_missile(
                                self.flying_missiles_dict[k]) < fleet_dict[fleet].distance:
                            if fleet_dict[fleet].channel_can_fire_count() >= N_0_residual:
                                fleet_dict[fleet].use_channel(N_0_residual)
                                break
                            else:
                                N_0_residual -= fleet_dict[fleet].channel_can_fire_count()
                                fleet_dict[fleet].use_channel(fleet_dict[fleet].channel_can_fire_count())

                P3 = 0.5

                P_3_list = P_3_k(self.flying_missiles_dict[k].numbers, n, self.flying_missiles_dict[k].P_3_input_list,
                                 P3)
                # print("P_3_list:{}".format(P_3_list))
                self.flying_missiles_dict[k].P_3_input_list = P_3_list
                self.flying_missiles_dict[k].update_coordinate(1, fleet_dict[self.flying_missiles_dict[k].target])



            else:
                if fleet_dict[self.flying_missiles_dict[k].target].dead == True :
                    # P_m = -100.0
                    # state = self.state
                    self.flying_missiles_dict[k].reward += -100
                    reward = -100-self.flying_missiles_dict[k].numbers
                    # self.state[-1] = step


                else:
                    P4 = 0.5
                    P_MJZ = 0.7
                    N_MJZ = 10
                    T_MJZ = 5
                    n_mijizhen = P_MJZ * N_MJZ * T_MJZ
                    P_4_list = list()
                    for i in range(self.flying_missiles_dict[k].numbers + 1):
                        P_4_list.append(P_4_i(self.flying_missiles_dict[k].numbers, i, n_mijizhen, self.flying_missiles_dict[k].P_3_input_list, P4))
                    self.flying_missiles_dict[k].P_4_list = P_4_list
                    # print("P_4_list:{}".format(P_4_list))

                    m = 1
                    self.flying_missiles_dict[k].P_m = P(m, self.flying_missiles_dict[k].numbers, self.flying_missiles_dict[k].P_4_list)
                    # print("P_m:{}".format(self.flying_missiles_dict[k].P_m))
                    # self.P_total[self.flying_missiles_dict[k].target][0] += self.flying_missiles_dict[k].P_m
                    # self.P_total[self.flying_missiles_dict[k].target][1] += self.flying_missiles_dict[k].numbers
                    # some_missiles_to_some_fleet_avg_pm = self.P_total[self.flying_missiles_dict[k].target][0]/self.P_total[self.flying_missiles_dict[k].target][1]
                    # print("some_missiles_to_some_fleet_avg_pm:{}, some_missiles_to_some_fleet_avg_pm>0.5:{}, coordinate :{},N_0:{},target:{}".
                    #       format(some_missiles_to_some_fleet_avg_pm,some_missiles_to_some_fleet_avg_pm>0.5,[self.flying_missiles_dict[k].x,self.flying_missiles_dict[k].y],
                    #              self.flying_missiles_dict[k].numbers,self.flying_missiles_dict[k].target))
                    if self.flying_missiles_dict[k].P_m > 0.1:
                        fleet_dict[self.flying_missiles_dict[k].target].died()
                        print("Is fleet {} died? Answer:{} .".format(self.flying_missiles_dict[k].target, fleet_dict[self.flying_missiles_dict[k].target].dead))

                        self.state[self.flying_missiles_dict[k].target] = 0
                        # next_state = self.state
                        # reward += self.P_total[self.flying_missiles_dict[k].target][0]/self.P_total[self.flying_missiles_dict[k].target][1] * 100 - self.flying_missiles_dict[k].numbers * 2
                        reward += self.flying_missiles_dict[k].P_m * 10 - self.flying_missiles_dict[k].numbers
                        self.flying_missiles_dict[k].reward += self.flying_missiles_dict[k].P_m * 10
                        # done_state = [0 for i in range(len(fleet_dict))]
                        if self.state[:self.state_dim] == self.done_state:
                            self.done = True
                            self.flying_missiles_dict[k].done = True
                            replayer_list.append([self.flying_missiles_dict[k].state, self.flying_missiles_dict[k].action,self.flying_missiles_dict[k].reward,
                                                  self.flying_missiles_dict[k].next_state,self.flying_missiles_dict[k].done,self.flying_missiles_dict[k].step,
                                                  self.flying_missiles_dict[k].episode,self.flying_missiles_dict[k].P_m,self.flying_missiles_dict[k].numbers])
                            will_be_pop_missile_list.append(k)
                            fire_channel_state = list()
                            for fleet in fleet_dict.values():
                                fire_channel_state.extend(fleet.fire_channels)
                                self.state[len(self.fleet_dict):] = fire_channel_state + [self.total_missiles]
                            for k in self.flying_missiles_dict.keys():
                                if self.flying_missiles_dict[k].update_next_state == True:
                                    self.flying_missiles_dict[k].next_state = self.state
                                    self.flying_missiles_dict[k].upate_next_state = False
                            for k in will_be_pop_missile_list:
                                self.flying_missiles_dict.pop(k)
                            for k in self.flying_missiles_dict.keys():
                                replayer_list.append([self.flying_missiles_dict[k].state, self.flying_missiles_dict[k].action,self.flying_missiles_dict[k].reward,
                                                      self.flying_missiles_dict[k].next_state,self.flying_missiles_dict[k].done,self.flying_missiles_dict[k].step,
                                                      self.flying_missiles_dict[k].episode,self.flying_missiles_dict[k].P_m,self.flying_missiles_dict[k].numbers])
                            return replayer_list,self.state
                        # info =
                        # return self.state, reward, self.done
                    else:
                        reward += -1.0 - self.flying_missiles_dict[k].numbers
                        self.flying_missiles_dict[k].reward += -1.0
                        # return self.state, reward, self.done
                replayer_list.append([self.flying_missiles_dict[k].state,self.flying_missiles_dict[k].action,self.flying_missiles_dict[k].reward,
                                      self.flying_missiles_dict[k].next_state,self.flying_missiles_dict[k].done,self.flying_missiles_dict[k].step,
                                      self.flying_missiles_dict[k].episode,self.flying_missiles_dict[k].P_m,self.flying_missiles_dict[k].numbers])
                will_be_pop_missile_list.append(k)
                    # self.flying_missiles_dict.pop(k)




        fire_channel_state = list()
        for fleet in fleet_dict.values():
            fire_channel_state.extend(fleet.fire_channels)
            self.state[len(self.fleet_dict):] = fire_channel_state + [self.total_missiles]
        # missile_a.next_state = self.state
        if missile_a.numbers == 0:
            missile_a.next_state = self.state
            # replayer_list.append([missile_a.state,missile_a.action,missile_a.reward,missile_a.next_state,missile_a.done,missile_a.step,missile_a.episode,missile_a.P_m,missile_a.numbers])
        else:
            self.flying_missiles_dict[step].next_state = self.state
        # for k in self.flying_missiles_dict.keys():
        #     if self.flying_missiles_dict[k].update_next_state == True:
        #         self.flying_missiles_dict[k].next_state = self.state
        #         self.flying_missiles_dict[k].upate_next_state = False
        for k in will_be_pop_missile_list:
            self.flying_missiles_dict.pop(k)
        return replayer_list,self.state


if __name__=="__main__":
    env=env()
    for step in range(100000):
        action = random.randint(0,160-1)
        print("action = {}, step = {}".format(action,step))
        print(env.step(action,step,4))




