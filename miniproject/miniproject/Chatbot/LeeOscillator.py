'''
    Copyright:      JarvisLee
    Date:           5/19/2021
    File Name:      LeeOscillator.py
    Description:    The Choatic activation functions named Lee-Oscillator Based on Raymond Lee's paper.
'''

# Import the necessary library.
import math
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


# Create the class for the Lee-Oscillator.
class LeeOscillator():
    '''
        The Lee-Oscillator based activation function.\n
        Params:\n
            - a (list), The parameters list for Lee-Oscillator of Tanh.\n
            - b (list), The parameters list for Lee-Oscillator of Sigmoid.\n
            - K (integer), The K coefficient of the Lee-Oscillator.\n
            - N (integer), The number of iterations of the Lee-Oscillator.\n
    '''

    # Create the constructor.
    def __init__(self, a1=5, a2=5, b1=1, b2=1, K=500, N=600, s=5, e=0.02, c=1, eu=0, ev=0, stepsize=0.002,
                 recordsize=100):
        # Get the parameters for the Lee-Oscillator.
        self.ev = ev
        self.eu = eu
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.K = K
        self.N = N
        self.s = s
        self.e = e
        self.c = c
        self.stepsize = stepsize
        self.recordsize = recordsize

        # Draw the bifraction diagram of the Lee-Oscillator.
        if (not os.path.exists('./LeeOscillator-Tanh_' + str(
                stepsize) + "_" + str(a1) + "-" + str(a2) + "-" + str(b1) + "-" + str(b2) + "-" + str(K) + "-" + str(eu) + "-" + str(ev) + '.csv')) or (
                not os.path.exists(
                    './LeeOscillator-Sigmoid_' + str(stepsize) + "_" + str(a1) + "-" + str(a2) + "-" + str(
                        b1) + "-" + str(b2) + "-" + str(K) + "-" + str(eu) + "-" + str(ev) + '.csv')):
            # Compute the Lee-Oscillator.
            print("LeeOscillator is not saved, prepare to save now...")
            self.Save()
            print("LeeOscillator is saved.")
        self.tanh = torch.tensor(
            pd.read_csv(
                './LeeOscillator-Tanh_' + str(stepsize) + "_" + str(a1) + "-" + str(a2) + "-" + str(b1) + "-" + str(
                    b2) + "-" + str(K) + "-" + str(eu) + "-" + str(ev) + '.csv', dtype=np.float32(), header=0,
                index_col=0).values)
        print(self.tanh)
        self.sigmoid = torch.tensor(
            pd.read_csv(
                './LeeOscillator-Sigmoid_' + str(stepsize) + "_" + str(a1) + "-" + str(a2) + "-" + str(b1) + "-" + str(
                    b2) + "-" + str(K) + "-" + str(eu) + "-" + str(ev) + '.csv', dtype=np.float32(), header=0,
                index_col=0).values)
        print("LeeOscillator is loaded.")

    def CheckVal(self, x):
        if x > -1 or x < 1:
            return torch.ones(1)
        else:
            return torch.zeros(1)

    # Create the function to apply the Lee-Oscillator of tanh activation function.
    def Tanh(self, x):
        output = torch.zeros(x.shape).to(x.device)
        # print(x)
        s = x.size()
        m = 100
        # print("m:", m, "xsize:", x.size())
        x = torch.reshape(x, (1, -1))
        cdt_1 = x > -1
        cdt_2 = x < 1
        # print("cdt_1: ", cdt_1)
        # temp_step = torch.tensor([[True], [False], [True], [True]]).cuda()
        idx = (cdt_1.int() * cdt_2.int()).bool()
        # print("idx: ", idx)
        # print("1 - idx: ", (1 - idx.int()).bool())
        # print("X (> -1, <1): ", torch.reshape(x * idx, s))
        # print("X (opsit): ", torch.reshape(x * (1 - idx.int()).bool(), s))
        x_idx = ((x + 1) / self.stepsize).int() * idx
        y_idx = torch.randint(0, m - 1, x.size()).to(x.device) * idx
        output_1 = self.TanhGet(x_idx, y_idx)
        output_2 = (1 - idx.int()).bool() * (1 / (1 + torch.exp(- x)))
        # output_2 = torch.where(torch.isnan(output_2), torch.full_like(output_2, 0), output_2)
        output = output_1 + output_2
        # print("output_1: ", output_1)
        # print("output_2: ", output_2)
        output = torch.reshape(output, s)
        # l = []
        # Get each value of the output.
        # for i in range(0, output.shape[0]):
        #     for j in range(0, output.shape[1]):
        #         if x[i][j] < -1 or x[i][j] > 1:
        #             output[i][j] = 1 / (1 + torch.exp(- x[i][j]))
        #         else:
        #             row = int((x[i][j] + 1) / self.stepsize)
        #             # l.append(row)
        #             col = int(np.random.rand() * (m - 1))
        #             output[i][j] = self.tanh[row][col]
        # cdt_1 = x > -1
        # cdt_2 = x < 1
        # # temp_step = torch.tensor([[True], [False], [True], [True]]).cuda()
        # idx = (cdt_1.int() * cdt_2.int())
        # # print((x < -1) + (x > 1)) rslt_1 = output + self.tanh[((x + 1) / self.stepsize).int().long() * idx.bool() ,
        # # int(np.random.rand() * (m - 1))] rslt_2 = 1 / (1 + torch.exp(- x)) * (1 - idx).bool() output = rslt_1 +
        # # rslt_2
        # # output = idx * (self.TanhGet(((x + 1) / self.stepsize), random.randint(0, m - 1))) + \
        # #          (idx * False) * (1 / (1 + torch.exp(- x)))
        # # output = torch.where(idx,
        # #                      self.TanhGet(((x + 1) / self.stepsize), random.randint(0, m - 1), idx),
        # #                      1 / (1 + torch.exp(- x)))
        # output = (self.TanhGet(((x + 1) / self.stepsize) * idx, torch.randint(0, m - 1, x.size()).to(x.device) * idx)) + (1 - idx) * torch.where(x <= -1, -0.9999, 0.9999)
        # # Return the output.
        # torch.set_printoptions(threshold=np.inf)
        # print("TanhGet: ", output)
        return Variable(output, requires_grad=True)

    #
    # def TanhGet(self, p, q):
    #     p = p.int().long()
    #     print("p: ", torch.reshape(p, (1, -1)), "q: ", torch.reshape(q, (1, -1)))
    #     return self.tanh[torch.reshape(p, (1, -1)), torch.reshape(q, (1, -1))]
    def TanhGet(self, p, q):
        p = p.long()
        # torch.set_printoptions(threshold=np.inf)
        # print("p: ", torch.reshape(p, (1, -1)), "q: ", torch.reshape(q, (1, -1)))
        return self.tanh[p, q]

    # Create the function to apply the Lee-Oscillator of sigmoid activation function.
    def Sigmoid(self, x):
        output = torch.zeros(x.shape).to(x.device)
        # print(x)
        s = x.size()
        # m = x.size(1)
        m = 100
        # print("m:", m, "xsize:", x.size())
        x = torch.reshape(x, (1, -1))
        cdt_1 = x > -1
        cdt_2 = x < 1
        # temp_step = torch.tensor([[True], [False], [True], [True]]).cuda()
        idx = (cdt_1.int() * cdt_2.int()).bool()
        # print("idx: ", idx)
        # print("1 - idx: ", (1 - idx.int()).bool())
        # print("X (> -1, <1): ", torch.reshape(x * idx, s))
        # print("X (opsit): ", torch.reshape(x * (1 - idx.int()).bool(), s))
        x_idx = ((x + 1) / self.stepsize).int() * idx
        y_idx = torch.randint(0, m - 1, x.size()).to(x.device) * idx
        output_1 = self.SigmoidGet(x_idx, y_idx)
        output_2 = (1 - idx.int()).bool() * (1 / (1 + torch.exp(- x)))
        # output_2 = torch.where(torch.isnan(output_2), torch.full_like(output_2, 0), output_2)
        output = output_1 + output_2
        # print("output_1: ", output_1)
        # print("output_2: ", output_2)
        output = torch.reshape(output, s)
        # l = []
        # Get each value of the output.
        # for i in range(0, output.shape[0]):
        #     for j in range(0, output.shape[1]):
        #         if x[i][j] < -1 or x[i][j] > 1:
        #             output[i][j] = 1 / (1 + torch.exp(- x[i][j]))
        #         else:
        #             row = int((x[i][j] + 1) / self.stepsize)
        #             # l.append(row)
        #             col = int(np.random.rand() * (m - 1))
        #             output[i][j] = self.tanh[row][col]
        # cdt_1 = x + 1 > 0
        #
        # cdt_2 = x - 1 < 0
        # # temp_step = torch.tensor([[True], [False], [True], [True]]).cuda()
        # idx = cdt_1.int() * cdt_2.int()
        # # print((x < -1) + (x > 1)) rslt_1 = output + self.tanh[((x + 1) / self.stepsize).int().long() * idx.bool() ,
        # # int(np.random.rand() * (m - 1))] rslt_2 = 1 / (1 + torch.exp(- x)) * (1 - idx).bool() output = rslt_1 +
        # # rslt_2
        # output = torch.where(idx.bool(),
        #                      self.SigmoidGet(((x + 1) / self.stepsize), random.randint(0, m - 1), idx.bool()),
        #                      1 / (1 + torch.exp(- x)))
        # # output = idx * (self.SigmoidGet(((x + 1) / self.stepsize), torch.randint(0, m - 1, x.size()))) + \
        # #          (idx * False) * (1 / (1 + torch.exp(- x)))
        # output = (self.SigmoidGet(((x + 1) / self.stepsize) * idx, torch.randint(0, m - 1, x.size()).to(x.device) * idx)) + (1 - idx) * torch.where(x <= -1, -0.9999, 0.9999)
        # Return the output.
        # print(l)
        # torch.set_printoptions(threshold=np.inf)
        # print("SigmoidGet: ", output)
        # return Variable(output, requires_grad=True)
        return Variable(output, requires_grad=True)

    def SigmoidGet(self, p, q):
        p = p.long()
        # torch.set_printoptions(threshold=np.inf)
        # print("p: ", torch.reshape(p, (1, -1)), "q: ", torch.reshape(q, (1, -1)))
        return self.sigmoid[p, q]

    # Create the function to compute the Lee-Oscillator of tanh activation function.
    def Save(self):
        idx = 0
        recbgnidx = self.N - self.recordsize
        recofst = (self.N - self.recordsize)
        xaixs = np.zeros([(int(2 / self.stepsize)) * self.recordsize], dtype=np.float32())
        xaxisidx = 0
        Z = np.zeros([int(2 / self.stepsize), self.recordsize], dtype=np.float32())
        u = torch.zeros([self.N])
        v = torch.zeros([self.N])
        z = torch.zeros([self.N])
        w = torch.zeros([self.N])
        z[0] = 0.2
        u[0] = 0.2
        for i in np.arange(-1, 1, self.stepsize):
            sign = 0
            if i != 0:
                if i > 0:
                    sign = 1
                if i < 0:
                    sign = -1
            it = i + 0.02 * sign
            for t in range(0, self.N - 1):
                tempu = self.a1 * u[t] - self.a2 * v[t] + it - self.eu
                tempv = self.b1 * u[t] - self.b2 * v[t] - self.ev
                u[t + 1] = torch.tanh(self.s * tempu)
                v[t + 1] = torch.tanh(self.s * tempv)
                w[t + 1] = torch.tanh(self.s * torch.Tensor([it]))
                z[t + 1] = ((u[t + 1] - v[t + 1]) * np.exp(-self.K * np.power(it, 2)) + self.c * w[t + 1])
                # print(z[t + 1])
                if t >= recbgnidx:
                    xaixs[xaxisidx] = i
                    xaxisidx = xaxisidx + 1
                    Z[idx, t - recofst] = z[t + 1]
                    # print("[", idx, ",", t - recofst, "]")
            Z[idx, t - recofst + 1] = z[t + 1]
            # print(t)
            idx = idx + 1
        # print("---------------------------------------------------")
        print(Z.dtype)
        data = pd.DataFrame(Z)
        data.to_csv('./LeeOscillator-Tanh_' + str(self.stepsize) + "_" + str(self.a1) + "-" + str(self.a2) + "-" + str(
            self.b1) + "-" + str(self.b2) + "-" + str(self.K) + "-" + str(self.eu) + "-" + str(self.ev) + '.csv')
        plt.figure(1)
        fig = np.reshape(Z, [(int(2 / self.stepsize)) * self.recordsize])
        plt.plot(xaixs, fig, ',')
        plt.savefig('./LeeOscillator-Tanh.jpg')
        plt.show()

        Z = Z / 2 + 0.5
        data = pd.DataFrame(Z)
        data.to_csv(
            './LeeOscillator-Sigmoid_' + str(self.stepsize) + "_" + str(self.a1) + "-" + str(self.a2) + "-" + str(
                self.b1) + "-" + str(self.b2) + "-" + str(self.K) + "-" + str(self.eu) + "-" + str(self.ev) + '.csv')
        plt.figure(1)
        fig = np.reshape(Z, [(int(2 / self.stepsize)) * self.recordsize])
        plt.plot(xaixs, fig, ',')
        plt.savefig('./LeeOscillator-Sigmoid.jpg')
        plt.show()


# Create the main function to test the Lee-Oscillator.
if __name__ == "__main__":
    # Get the parameters list of the Lee-Oscillator.
    # Create the Lee-Oscillator's model.
    # Lee = LeeOscillator(a, b, 50, 100)
    Lee1 = LeeOscillator()
    # Lee2 = LeeOscillator(a1=5, a2=5, b1=5, b2=5, eu=1, ev=1)
    # Lee.Compute()
    # Test the Lee-Oscillator.

