import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):
 
        SE = SE.unsqueeze(0).unsqueeze(0).cuda()
        SE = self.FC_se(SE)  # SE(1, 1,  64)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # (bs, his+pre, 295)
        TE = TE.unsqueeze(dim=2).cuda()  # (bs, his+pre, 1, 295)
        TE = self.FC_te(TE)  # (bs, his+pre, 1, 64)
        del dayofweek, timeofday
        return SE + TE


class Attention(nn.Module):

    def __init__(self, K, d, bn_decay, mask=True):
        super(Attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step,  d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        # query: [K * batch_size,num_step, d]
        # key:   [K * batch_size,  d, num_step]
        # value: [K * batch_size, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        # [K * batch_size,  num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        # softmax
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X



class MAN(nn.Module):

    def __init__(self, SE, args, bn_decay):
        super(MAN, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        self.num_his = args.num_his
        self.SE = SE
        self.STEmbedding = STEmbedding(D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([Attention(K, d, bn_decay) for _ in range(L)])

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, X, TE):

        X = torch.unsqueeze(X, -1)
        X = self.FC_1(X)

        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]

        for net in self.STAttBlock_1:
            X = net(X, STE_his)

        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)


# Creating the architecture of the Neural Network, Stacked Auto Encoder that is going to be the child class of an
# existing parent class in pytorch called Module M which is taken from the nn module that we import there
class SAE(nn.Module):
    # since we are doing an inheritance, the arguement need not be added, but need ,
    def __init__(self, in_dim, out_dim):
        # use super() function to optimize the SAE
        super(SAE, self).__init__()
   
        self.fc1 = nn.Linear(in_dim, in_dim-1)   
        # the second full connection, the first feature will be 20 that is the hidden nodes of the first hidden
        # layer,  10 is determined as the size of second hidden layer, this will detect more features based on the
        # first hidden layer
        self.fc2 = nn.Linear(in_dim-1, in_dim-2)
        self.fc3 = nn.Linear(in_dim-2, in_dim-3)
        # start to decode or reconstruct the original input vector, the second feature will equal to the firs feature
        # of fc2
        self.fc4 = nn.Linear(in_dim-3, in_dim-2)
        self.fc5 = nn.Linear(in_dim-2, in_dim-1)
        # same reason, for the reconstruction of the input vector, the output vetor should have the same dimention as
        # the input vector
        self.fc6 = nn.Linear(in_dim-1, out_dim)  
        # determin the activation function, sigmoid or you can use other activation function to compare
        self.activation = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        if type(self.fc1) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)
        if type(self.fc2) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)
        if type(self.fc3) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)
        if type(self.fc4) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)
        if type(self.fc5) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)
        if type(self.fc6) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)


    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))

  
        x = self.fc6(x)
        return x


class STE_SAE(nn.Module):

    def __init__(self, D, bn_decay, in_dim, out_dim):
        super(STE_SAE, self).__init__()
        self.SAE = SAE(in_dim, out_dim)

        self.FC_pe = FC(
            input_dims=[1, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

        self.FC_te = FC(
            input_dims=[295, out_dim], units=[out_dim, out_dim], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, FE, T=288):
        PE = FE[:, :107]  # (bs, 107)
        TE = FE[:, 107:109]
        SE = FE[:, 109:]

        # position embedding
        PE = PE.unsqueeze(dim=1).unsqueeze(dim=-1).cuda()  # bs, 1, 107, 1
        PE = self.FC_pe(PE)  # bs, 1, 107, 64

        # feature embedding
        SE = SE.cuda()
        SE = self.SAE(SE)   # bs, out_dim
        SE = SE.unsqueeze(dim=1).unsqueeze(dim=2)  # bs, 1, 1, out_dim

        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], 7)  # bs,7
        timeofday = torch.empty(TE.shape[0], T)  # bs,288
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # bs, 295
        TE = TE.unsqueeze(dim=1).unsqueeze(dim=2).cuda() # bs, 1, 1, 295
        TE = self.FC_te(TE)  # bs, 1, 1, 295 -> # bs, 1, 1, 9
        del dayofweek, timeofday
        return PE, SE + TE


class Mix_MAN(nn.Module):

    def __init__(self, SE, args, bn_decay, in_dim, out_dim):
        super(Mix_MAN, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        self.num_his = args.num_his
        self.SE = SE
        self.STEmbedding = STEmbedding(D, bn_decay)

        self.STE_SAE = STE_SAE(D, bn_decay, in_dim, out_dim)

        self.STAttBlock_1 = nn.ModuleList([Attention(K, d, bn_decay) for _ in range(L)])

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

        self.FC_3 = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_4 = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)

        self.FC_s = FC(input_dims=[self.num_his+out_dim, self.num_his], units=[self.num_his, self.num_his], activations=[F.relu, None],
                       bn_decay=bn_decay)

        self._initialize_weights()

    def _initialize_weights(self):
        if type(self.FC_s) == nn.Linear:
            nn.init.normal_(self.fc4.weight, std=0.01)

    def forward(self, X, TE, FE):

        X = torch.unsqueeze(X, -1)
        X = self.FC_1(X)  
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]     
        STE_pred = STE[:, self.num_his:]    

        x1 = self.STAttBlock_1(X, STE_his)

        X4 = self.FC_3(X  + x1)
        X = X + X4

        FE = FE.unsqueeze(dim=1).unsqueeze(dim=2).expand(-1, X.shape[1], X.shape[2], -1)
        X = torch.cat([X, FE], dim=-1)
        X = self.FC_2(X)    

        PE, FE = self.STE_SAE(FE)  
        PE = PE.expand(-1, X.shape[1], -1, -1)
        X = X + X*PE
        X = self.FC_2(X)  

        X = X.permute(0, 3, 2, 1)
        FE = FE.expand(-1, X.shape[1], X.shape[2], -1)
        X = torch.cat([X, FE], dim=-1)
        X = self.FC_s(X)
        X = X.permute(0, 3, 2, 1)

        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)


