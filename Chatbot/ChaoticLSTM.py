'''
    Copyright:      JarvisLee
    Date:           5/1/2021
    File Name:      ChaoticLSTM.py
    Description:    The Chaotic based Long-Short Term Memory Unit.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import numpy as np
from LeeOscillator import LeeOscillator


# Create the class for the Chaotic Long-Short Term Memory Unit.
class ChaoticLSTM(nn.Module):
    '''
        The Chaotic Long-Short Term Memory Unit.\n
        Params:\n
            - inputSize (integer), The input size of the Chaotic LSTM.\n
            - hiddenSize (integer), The output size of the Chaotic LSTM.\n
            - Lee (LeeOscillator), The Lee-Oscillator.\n
            - chaotic (bool), The boolean to check whether use the Chaotic Mode.\n
            - bidirection (bool),  The boolean to check whether apply the Bi-LSTM.\n
    '''

    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, Lee=None, chaotic=False, bidirection=False):
        # Create the super constructor.
        super(ChaoticLSTM, self).__init__()
        # Get the member variables.
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.chaotic = chaotic
        self.bidirection = bidirection
        # Create the parameter of the input.
        if chaotic:
            mut = 2
        else:
            mut = 4
        self.Wi = nn.Parameter(torch.Tensor(inputSize, hiddenSize * mut)).cuda()
        # Create the parameter of the hidden.
        self.Wh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * mut)).cuda()
        # Create the parameter of the bias.
        self.B = nn.Parameter(torch.Tensor(hiddenSize * mut)).cuda()
        # Check the number of directions.
        if bidirection == True:
            # Create the parameter of the inverse input.
            self.Winvi = nn.Parameter(torch.Tensor(inputSize, hiddenSize * mut)).cuda()
            # Create the parameter of the inverse hidden.
            self.Winvh = nn.Parameter(torch.Tensor(hiddenSize, hiddenSize * mut)).cuda()
            # Create the parameter of the inverse bias.
            self.Binv = nn.Parameter(torch.Tensor(hiddenSize * mut)).cuda()
        # Initialize the parameters.
        self.initParams()
        # Create the chaotic activation function.
        self.Lee = Lee

    # Initialize the parameters.
    def initParams(self):
        # Compute the standard deviation.
        std = 1.0 / np.sqrt(self.hiddenSize)
        # Initialize the parameters.
        for param in self.parameters():
            param.data.uniform_(-std, std)

    # Create the forward propagation.
    def forward(self, x, initStates=None, last_bi=False):
        # x = torch.randn((32, 10, 46))
        # Get the batch size and sequence size.
        # bs, seqs, _ = x.size()
        seqs, bs, _ = x.size()
        # Create the list to store the output.
        output = []
        # Initialize the hidden, cell, inverse hidden and inverse cell.
        if initStates is None:
            if self.bidirection == True:
                ht, ct, hinvt, cinvt = (
                    torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device),
                    torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
            else:
                ht, ct = (torch.zeros(bs, self.hiddenSize).to(x.device), torch.zeros(bs, self.hiddenSize).to(x.device))
        else:
            if self.bidirection == True:
                ht, ct, hinvt, cinvt  = initStates
                # [ht, hinvt] = initStates
            else:
                # ht = initStates[-(1 + int(self.last_bi)):]
                # ht = torch.cat(ht.split(1), dim=-1).squeeze(0)
                # initStates = initStates[-(1 + int(False)):]
                # (ht, ct) = torch.cat(initStates.split(1), dim=-1).squeeze(0)
                ht, ct = initStates
        # Compute the LSTM.
        for t in range(seqs):
            # Get the xt.
            # xt = x[:, t, :]
            xt = x[t, :, :]
            # print("xt: ", xt)
            # Compute the gates.
            gates = xt @ self.Wi + ht @ self.Wh + self.B
            # print("xt @ self.Wi: ", xt @ self.Wi)
            # Get the value of the output.
            if self.chaotic == True:
                it, ft= (
                    self.Lee.Sigmoid(gates[:, :self.hiddenSize]).to(x.device),
                    self.Lee.Sigmoid(gates[:, self.hiddenSize:self.hiddenSize * 2]),
                )
                # print("it: ", it)
                # Compute the cell and hidden.
                ot = self.Lee.Sigmoid(xt)
                ct = ft * ct + it
                ht = ot * self.Lee.Tanh(ct)
                # print("ht: ", ht)
            else:
                it, ft, gt, ot = (
                    torch.sigmoid(gates[:, :self.hiddenSize]).to(x.device),
                    torch.sigmoid(gates[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device),
                    torch.tanh(gates[:, self.hiddenSize * 2:self.hiddenSize * 3]).to(x.device),
                    torch.sigmoid(gates[:, self.hiddenSize * 3:]).to(x.device)
                )
                # Compute the cell and hidden.
                ct = ft * ct + it * gt
                ht = ot * torch.tanh(ct).to(x.device)
            # Store the forward value.
            output.append(ht.unsqueeze(0))
        # Check the direction.
        if self.bidirection == True:
            for t in range(seqs - 1, -1, -1):
                # Get the xinvt.
                # xinvt = x[:, t, :]
                xinvt = x[t, :, :]
                # print("xinvt: ", xinvt)
                # Compute the inverse gates
                gatesInv = xinvt @ self.Winvi + hinvt @ self.Winvh + self.Binv
                # print("ingate: ", gatesInv)
                # Get the value of the output.
                if self.chaotic == True:
                    # Get the value of each inverse gate.
                    iinvt, finvt = (
                        self.Lee.Sigmoid(gatesInv[:, :self.hiddenSize]).to(x.device),
                        self.Lee.Sigmoid(gatesInv[:, self.hiddenSize:self.hiddenSize * 2])
                    )
                    # print("iinvt: ", iinvt)
                    # Compute the inverse cell and hidden.
                    oinvt = self.Lee.Sigmoid(xinvt)
                    cinvt = finvt * cinvt + iinvt
                    hinvt = oinvt * self.Lee.Tanh(cinvt)
                    # print("it: ", it)
                    # Compute the cell and hidden.
                    # print("hinvt: ", hinvt)
                else:
                    # Get the value of each inverse gate.
                    iinvt, finvt, ginvt, oinvt = (
                        torch.sigmoid(gatesInv[:, :self.hiddenSize]).to(x.device),
                        torch.sigmoid(gatesInv[:, self.hiddenSize:self.hiddenSize * 2]).to(x.device),
                        torch.tanh(gatesInv[:, self.hiddenSize * 2:self.hiddenSize * 3]).to(x.device),
                        torch.sigmoid(gatesInv[:, self.hiddenSize * 3:]).to(x.device)
                    )
                    # Compute the inverse cell and hidden.
                    cinvt = finvt * cinvt + iinvt * ginvt
                    hinvt = oinvt * torch.tanh(cinvt).to(x.device)
                # Store the backward value.
                output[t] = torch.cat([output[t], hinvt.unsqueeze(0)], dim=2)
            # Concatenate the hidden and cell.
            # ht = torch.cat([ht, hinvt], dim=1)
            ht = torch.cat([ht.unsqueeze(0), hinvt.unsqueeze(0)], dim=0)
            ct = torch.cat([ct.unsqueeze(0), cinvt.unsqueeze(0)], dim=0)
            # ct = torch.cat([ct, cinvt], dim=1)
            # Concatenate the output, hidden and cell.
        output = torch.cat(output, dim=0)
        # Return the output, hidden and cell.
        return output, (ht, ct)


# Create the main function to test the Chaotic LSTM.
if __name__ == "__main__":
    # Get the Lee-Oscillator.
    Lee = LeeOscillator()
    # Create the Bi-LSTM unit.
    # CLSTM = ChaoticLSTM(inputSize=46, hiddenSize=10, Lee=Lee, chaotic=False, bidirection=True)
    # # Test the Bi-LSTM.
    # x = torch.randn((32, 10, 46))
    # print(x.shape)
    # output, (h, c) = CLSTM(x)
    # print(output.shape)
    # print(h.shape)
    # print(c.shape)
    #
    # # Create the Chaotic Bi-LSTM unit.
    # CLSTM = ChaoticLSTM(inputSize=46, hiddenSize=10, Lee=Lee, chaotic=True, bidirection=True)
    # # Test the Chaotic Bi-LSTM.
    # x = torch.randn((32, 10, 46))
    # print(x.shape)
    # output, (h, c) = CLSTM(x)
    # print(output.shape)
    # print(h.shape)
    # print(c.shape)

    # Create the Chaotic LSTM unit.
    CLSTM = ChaoticLSTM(inputSize=1536, hiddenSize=1536, Lee=Lee, chaotic=True, bidirection=True)
    # Test the Chaotic LSTM.
    x = torch.randn((10, 128, 1536)).cuda()
    print(x.shape)
    output, (h, c) = CLSTM(x)
    print(output)
    print(h)
    print(c)

    # # Create the LSTM unit.
    # CLSTM = ChaoticLSTM(inputSize=46, hiddenSize=46, Lee=Lee, chaotic=False, bidirection=False)
    # # Test the LSTM.
    # x = torch.randn((10, 32, 46)).cuda()
    # print(x.shape)
    # output, (h, c) = CLSTM(x)
    # print(output)
    # print(h)
    # print(c)
