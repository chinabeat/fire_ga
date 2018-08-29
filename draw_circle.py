# _*_ coding=utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
import math


class fleet(object):
    # fire_num:Number of fire channels
    # distance:Intercept distance , km unit
    # time:Fire channel occupancy time , s unit
    # x:x Axis coordinate , km unit
    # y:y axis coordinate , km unit
    def __init__(self, fire_num, distance, oc_time, x, y):
        self.fire_num = fire_num
        self.distance = distance
        self.oc_time = oc_time
        self.x = x
        self.y = y
        # self.res_fire_num=fire_num

    # the distance between the fleet and missile
    # x : x axis coordinate of missile
    # y : y axis coordinate of missile
    def distance_to_missile(self, missile):
        return math.sqrt((missile.x - self.x) ** 2 + (missile.y - self.y) ** 2)

    def occupy_fire_channel(self):
        self.fire_num -= 1

    def relieve_fire_channel(self):
        self.fire_num += 1
def draw_circle(r,a,b,axes):
    # ==========================================
    # 圆的基本信息
    # 1.圆半径
    r = r
    # 2.圆心坐标
    a, b = (a, b)

    # ==========================================
    # 方法一：参数方程
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)


    # plt.plot(x, y)
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    axes.plot(x, y)

    axes.axis('equal')

    # # ==========================================
    # # 方法二：标准方程
    # x = np.arange(a-r, a+r, 0.01)
    # y = b + np.sqrt(r**2 - (x - a)**2)
    #
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # axes.plot(x, y) # 上半部
    # axes.plot(x, -y) # 下半部
    #
    # plt.axis('equal')

    # ==========================================

if __name__=="__main__":
    fleet_a = fleet(8, 20, 20, 18, 8 + 6 * math.sqrt(3))
    fleet_b = fleet(8, 20, 20, 24, 8 + 4 * math.sqrt(3))
    fleet_c = fleet(4, 15, 25, 12, 8 + 4 * math.sqrt(3))
    fleet_d = fleet(4, 15, 25, 30, 8 + 2 * math.sqrt(3))
    fleet_e = fleet(4, 15, 25, 6, 8 + 2 * math.sqrt(3))
    fleet_f = fleet(4, 15, 25, 36, 8)
    fleet_g = fleet(4, 15, 25, 0, 8)
    fleet_h = fleet(4, 15, 25, 26, 0)
    fleet_i = fleet(4, 15, 25, 10, 0)
    fleet_j = fleet(4, 15, 25, 18, 8)
    fleet_k = fleet(4, 34, 20, 18, 10)

    fleet_dict = {"a": fleet_a,
                  "b": fleet_b,
                  "c": fleet_c,
                  "d": fleet_d,
                  "e": fleet_e,
                  "f": fleet_f,
                  "g": fleet_g,
                  "h": fleet_h,
                  "i": fleet_i,
                  "j": fleet_j,
                  "k": fleet_k
                  }
    fig = plt.figure()
    axes = fig.add_subplot(111)
    for fleet in fleet_dict.values():
        draw_circle(fleet.distance,fleet.x,fleet.y,axes)

    plt.show()