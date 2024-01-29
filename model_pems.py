import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, dense_diff_pool, models
import math
import copy

class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.all_head_size,)
        context = context.view(*new_size)
        return context


class DotProductAttention(nn.Module):
    '''缩短点积注意力'''

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # queries的形状：(batch_size，查询的个数，d)
        # keys的形状：(batch_size，“键－值”对的个数，d)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.Softmax(scores)
        return torch.bmm(self.dropout(self.attention_weights), values)


class Interacte_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Interacte_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        device_gru = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 设置可以训练的参数矩阵
        self.w_xr = torch.nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_hr = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_cr = torch.nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_xz = torch.nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_hz = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_cz = torch.nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_xh = torch.nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_hh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device_gru))
        self.w_ch = torch.nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device_gru))

        self.b_r = torch.nn.Parameter(torch.randn(hidden_size, requires_grad=True, device=device_gru))
        self.b_z = torch.nn.Parameter(torch.randn(hidden_size, requires_grad=True, device=device_gru))
        self.b_h = torch.nn.Parameter(torch.randn(hidden_size, requires_grad=True, device=device_gru))

        self.outlinear = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()   #.Sigmoid()
        )  # 输出层

        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, input_c):
        batch_size = input.size(0)  # 一个batch的大小
        step_size = input.size(1)  # 时间步

        # 初始化隐藏状态矩阵h为零矩阵
        # h = torch.zeros(batch_size, self.hidden_size).cuda()
        h = torch.rand(batch_size, self.hidden_size).cuda()

        # 这里面存放每一个时间步出来的h
        lisths = []

        # 一个时间步一个时间步的计算
        for i in range(step_size):
            # 取input每个时间步的数据
            x = input[:, i, :]
            c = input_c[:, i, :]
            # --------------------------------GRU核心公式-----------------------------------
            # x形状是(batchsize,inputsize),w_xz矩阵形状是(inputsize,hiddensize)
            # torch.mm是矩阵乘法，这样(torch.mm(x,self.w_xz)的形状是(batchsize,hiddensize)

            z = torch.sigmoid((torch.mm(x, self.w_xz) + torch.mm(h, self.w_hz) + torch.mm(c, self.w_cz) + self.b_z))
            r = torch.sigmoid((torch.mm(x, self.w_xr) + torch.mm(h, self.w_hr) + torch.mm(c, self.w_cr) + self.b_r))
            h_tilde = torch.tanh(
                (torch.mm(x, self.w_xh) + torch.mm(r * h, self.w_hh) + torch.mm(c, self.w_ch) + self.b_h))
            h = (1 - z) * h + z * h_tilde

            # --------------------------------GRU核心公式-----------------------------------
            # h的形状是(batch_size,hidden_size)

            # 把每个时间步出来的h都存到list里
            lisths.append(h)

        # 用torch.stack把装有tensor的list转为torch.tensor类型,dim=1是指从第二个维度转化，因为seqlen在第二个维度上
        # 所以hs的形状是(batchsize,seqlen,hiddensize)
        hs = torch.stack(lisths, dim=1)  # 全部cell所计算的隐藏状态的集合

        # 此时的h是最后一个时间步计算出来的h，可以用这个作为最后的输出
        output = self.outlinear(h)  # 最后一个单元的输出，接一个带有Sigmoid激活函数的线性层，因为我们的任务是分类任务
        # output矩阵的形状现在是(batchsize,outputsize)

        output = F.relu(output)
        return output, hs

class densediff_pool(nn.Module):
    def __init__(self, num_node_in, num_node_out):
        super(densediff_pool, self).__init__()
        self.lin_ass = nn.Linear(num_node_out, num_node_out)

    def forward(self, x, s, mask=None, normalize=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        s = s.unsqueeze(0) if s.dim() == 2 else s

        batch_size, num_nodes, _ = x.size()

        s = torch.softmax(s, dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        assign = torch.matmul(x, s)
        assign = self.lin_ass(assign)
        # out = torch.matmul(x, assign)

        # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        #
        # link_loss = adj - torch.matmul(s, s.transpose(1, 2))
        # link_loss = torch.norm(link_loss, p=2)
        # if normalize is True:
        #     link_loss = link_loss / adj.numel()
        #
        # EPS = 1e-15
        # ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

        return assign



class GraphInteraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphInteraction, self).__init__()
        device_inter = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Message
        self.lin_self_input = nn.Linear(in_channels, in_channels)
        self.lin_self_adj_ped7 = torch.nn.Parameter(torch.randn(228, 228, requires_grad=True, device=device_inter))   # nn.Linear(228, 228)
        self.lin_self_adj_metr = torch.nn.Parameter(torch.randn(207, 207, requires_grad=True, device=device_inter))   # nn.Linear(207, 207)
        self.lin_self_input2 = nn.Linear(in_channels, out_channels, bias=False)
        # self.lin_self_input3 = nn.Linear(out_channels, 1)
        # self.lin_self_graph = nn.Linear(out_channels, out_channels)
        # self.lin_other_graph = nn.Linear(out_channels, in_channels)
        self.lin_other_graph = nn.Sequential(
                                    nn.Linear(in_channels, 27),
                                    nn.Linear(27, 12),
                                    nn.ReLU())
        self.att_self_graph = selfAttention(1, in_channels, out_channels)
        # self.att_other_graph = selfAttention(1, in_channels, out_channels)

        # self.gnn_mlp = models.MLP(in_channels=228, hidden_channels=166, out_channels=207, num_layers=3)
        self.mlp_for_pe7 = nn.Linear(in_channels, 207)    # inputsize is the number nodes of pemsd7
        self.mlp_for_metr = nn.Linear(in_channels, 228)

        self.dense_diff_pool_pe7metr = densediff_pool(num_node_in=228, num_node_out=207)
        self.dense_diff_pool_metrpe7 = densediff_pool(num_node_in=207, num_node_out=228)
        self.w_c_1 = nn.Linear(in_channels, in_channels)
        # self.w_c_2 = nn.Linear(in_channels, out_channels)
        # self.w_c_3 = nn.Linear(in_channels, out_channels)

        # Aggregate
        # self.inte_gru = Interacte_GRU(input_size=out_channels, hidden_size=128, output_size=out_channels)
        self.inte_gru_ped7 = Interacte_GRU(input_size=228, hidden_size=128, output_size=228)
        self.inte_gru_metr = Interacte_GRU(input_size=207, hidden_size=128, output_size=207)

        self.inte_lin = nn.Sequential(
                nn.Linear(out_channels, 37),
                nn.Linear(37, 12),
                nn.ReLU())
        self.inte_torchgru = nn.GRU(12, 1, batch_first=True)

        #
        self.res_lin = nn.Sequential(
                nn.Linear(in_channels, 27),
                nn.Linear(27, 12),
                nn.ReLU(),
                nn.Linear(12, 1))

    def forward(self, x, c, adj):
        return self.propagate(x, c, adj)

    def propagate(self, x, c, adj):
        out = self.message(x, c, adj)
        out = self.aggregate(out[0], out[1])

        out = self.res_lin(x.transpose(1,2)).transpose(1,2) + out
        out = self.update(out)
        return out

    def message(self, x, x_c_list, adj):

        # m = self.inte_lin(x.transpose(1, 2))
        # m = m.transpose(1, 2)

        degree_matrix = torch.sum(adj, dim=1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag_embed(degree_matrix)
        d_a = torch.matmul(degree_matrix, adj)
        if d_a.size()[0] == 228:
            d_a = self.lin_self_adj_ped7*(d_a)
        elif d_a.size()[0] == 207:
            d_a = self.lin_self_adj_metr*(d_a)

        x_lin = self.lin_self_input(x.transpose(1,2))
        self_att_H = self.att_self_graph(x.transpose(1,2))
        m = self.lin_self_input2(torch.matmul(d_a, x_lin)) + self_att_H

        m = self.inte_lin(m)
        m = m.transpose(1, 2)

        node_m = x.shape[-1]
        out_c_list = []
        for x_c in x_c_list:
            node_c = x_c.shape[-1]
            if node_c > node_m:
                assignment_matrix_c = self.mlp_for_pe7(x_c.transpose(1,2))  # 可以更改为其他池化方法
                out_c = self.dense_diff_pool_pe7metr(x=x_c, s=assignment_matrix_c)
            elif node_c < node_m:
                assignment_matrix_c = self.mlp_for_metr(x_c.transpose(1,2))
                out_c = self.dense_diff_pool_metrpe7(x=x_c, s=assignment_matrix_c)
            else:
                out_c = x_c
            out_c_list.append(out_c)

        # out_c_act = self.w_c_1(out_c_list[0]) + self.w_c_2(out_c_list[1])
        out_c_act = self.w_c_1(out_c_list[0].transpose(1,2))
        # other_att_H = self.att_other_graph(out_c_list[0].transpose(1,2))
        # out_c_act = F.relu(other_att_H)
        c = self.lin_other_graph(out_c_act).transpose(1,2)

        # c = F.dropout(c, p=0.3, training=self.training)
        # m = F.dropout(m, p=0.3, training=self.training)
        m = F.relu(m + x)
        c = F.relu(c + x)

        return m, c

    def aggregate(self, m, c):
        # aggr_out = self.inte_gru(m, c)
        if m.size()[2] == 228:
            aggr_out = self.inte_gru_ped7(m, c)
            aggr_out = aggr_out.unsqueeze(1)
        elif m.size()[2] == 207:
            aggr_out = self.inte_gru_metr(m, c)
            aggr_out = aggr_out.unsqueeze(1)
        else:
            raise ValueError("无法合并")
        # aggr_out,_ = self.inte_lin(m.transpose(1, 2))
        # aggr_out = aggr_out.transpose(1, 2)
        # return aggr_out
        return aggr_out

    def update(self, aggr_out):
        return aggr_out


# 原版
# class MultiGraphInteraction(nn.Module):
#     def __init__(self, in_channels, hiden_channels, att_channels, node_list, task_list):
#         '''
#         in_channels = 12 (Time Steps)
#         hiden_channels = [24, 24, 24, 24]  (Hidensize for every dataset)
#         att_channels = [36, 24, 68, 21]    (Hidensize in self-att module, which also for every dataset)
#         task_list = ['pems03', 'pems04', 'pems07', 'pems08']
#         '''
#         super(MultiGraphInteraction, self).__init__()
#         device_inter = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.task_list = task_list
#
#         # Message
#         self.inputs_lin = nn.ModuleDict({task_list[i]: nn.Linear(in_channels, hiden_channels[i]) for i in range(len(task_list))})
#         self.att_self_graph = nn.ModuleDict({task_list[i]: selfAttention(1, in_channels, att_channels[i]) for i in range(len(task_list))})
#         self.aggregate_lin = nn.ModuleDict({task_list[i]: nn.Linear(hiden_channels[i], att_channels[i]) for i in range(len(task_list))})
#         self.rec_T = nn.ModuleDict({task_list[i]: nn.Linear(att_channels[i], in_channels) for i in range(len(task_list))})
#
#         att_act_dict = {}
#         for i, task in enumerate(task_list):
#             act_task_set = copy.deepcopy(task_list)
#             act_task_set.pop(i)
#             num_node = node_list[i]
#             agg_nodes = copy.deepcopy(node_list)
#             agg_nodes.pop(i)
#             att_act_dict[task] = [zip(act_task_set, agg_nodes), num_node]
#         self.att_act = nn.ModuleDict({task: nn.ModuleDict({act_task: selfAttention(1, agg_nodes, act_task_set[1]) for act_task, agg_nodes in act_task_set[0]}) for task,act_task_set in att_act_dict.items()})
#
#
#         # self.att_act = {}
#         # for i, task in enumerate(task_list):
#         #     act_task_set = copy.deepcopy(task_list)
#         #     act_task_set.pop(i)
#         #     num_node = node_list[i]
#         #     agg_nodes = copy.deepcopy(node_list)
#         #     agg_nodes.pop(i)
#         #     # att_act_dict[task] = [zip(act_task_set, agg_nodes), num_node]
#         #     self.att_act[task] = nn.ModuleDict({act_task: selfAttention(1, agg_nodes[j], num_node) for j, act_task in enumerate(act_task_set)})
#         #     # for j, act_task in enumerate(act_task_set):
#         #     #     self.att_act[task][act_task] = nn.ModuleDict({act_task: selfAttention(1, agg_nodes[j], num_node)})
#         self.fusion_act = nn.ModuleDict({task_list[i]: nn.Linear(3*node_list[i], node_list[i], bias=False) for i in range(len(task_list))})
#
#         # Aggreate
#         self.aggr_GRU = nn.ModuleDict({task_list[i]: Interacte_GRU(node_list[i], int(node_list[i]/2), node_list[i]) for i in range(len(task_list))})
#
#         # Update
#         self.res_lin = nn.ModuleDict({task_list[i]: nn.Sequential(
#                 nn.Linear(in_channels, 27),
#                 nn.Linear(27, 12),
#                 nn.ReLU(),
#                 nn.Linear(12, 1)) for i in range(len(task_list))})
#
#     def forward(self, x_list, adj_list):
#         return self.propagate(x_list, adj_list)
#
#     def propagate(self, x_list, adj_list):
#         M, C = self.message(x_list, adj_list)
#         out = self.aggregate(M, C)
#         out = self.update(out, x_list)
#         return out
#
#     def message(self, x_list, adj_list):
#         task_set = list(x_list.keys())
#         aggregations = []
#         for task in task_set:
#             degree_matrix = torch.sum(adj_list[task], dim=1, keepdim=False)
#             degree_matrix = degree_matrix.pow(-1)
#             degree_matrix[degree_matrix == float("inf")] = 0.
#             degree_matrix = torch.diag_embed(degree_matrix)
#             d_a = torch.matmul(degree_matrix, adj_list[task])
#
#             x_lin = self.inputs_lin[task](x_list[task].transpose(1,2))                  # [B, T, N] —> [B, N, H1]
#             self_att_H = self.att_self_graph[task](x_list[task].transpose(1,2))         # [B, H1, N] —> [B, N, Att1]
#             m = self.aggregate_lin[task](torch.matmul(d_a, x_lin)) + self_att_H         # [B, N, Att1]
#             m = self.rec_T[task](m)                                                     # [B, N, Att1] —> [B, N, T]
#             m = m.transpose(1, 2)                                                       # [B, N, T] —> [B, T, N]
#             aggregations.append(m)
#
#         act_tensor_list = []
#         for i, task in enumerate(task_set):
#             act_task_set = copy.deepcopy(task_set)
#             act_task_set.pop(i)
#             act_tensors = []
#             for j, act_task in enumerate(act_task_set):
#                 agg_infor = self.att_act[task][act_task](x_list[act_task])              # [B, T, Nj] —> [B, T, Ni]
#                 act_tensors.append(agg_infor)
#             act_tensor = torch.cat(act_tensors, dim=2)                                  # [B, T, 3*Ni]
#             act_tensor = self.fusion_act[task](act_tensor)                              # [B, T, Ni]
#             act_tensor_list.append(act_tensor)
#
#         for i, task in enumerate(task_set):
#             aggregations[i] = F.relu(aggregations[i] + x_list[task])
#
#         return aggregations, act_tensor_list
#
#     def aggregate(self, m, c):
#         # aggr_out = self.inte_gru(m, c)
#         aggr_list = []
#         for i, task in enumerate(self.task_list):
#             aggr_out = self.aggr_GRU[task](m[i], c[i])
#             aggr_out = aggr_out.unsqueeze(1)
#             aggr_list.append(aggr_out)
#
#         return aggr_list
#
#     def update(self, aggr_out, x_list):
#         update_out_list = []
#         for i, task in enumerate(self.task_list):
#             update_out = self.res_lin[task](x_list[task].transpose(1, 2)).transpose(1, 2) + aggr_out[i]
#             update_out_list.append(update_out)
#         return update_out_list

class AEattention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super(AEattention, self).__init__()
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = torch.nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / dim_k**0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: batch, n, dim_in
        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = self.softmax(dist)  # batch, n, n
        att = torch.bmm(dist, v)
        return att, dist

class Adap_attention(nn.Module):
    '''
    非临接mask矩阵的注意力
    '''
    def __init__(self, num_nodes, dim_in, dim_v):
        super(Adap_attention, self).__init__()
        device_gru = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.zeros([num_nodes, num_nodes], device=device_gru, requires_grad=True)
        self.mask = nn.Parameter(mask)
        nn.init.xavier_normal_(self.mask, gain=0.0003)

        self.linear_v = torch.nn.Linear(dim_in, dim_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        adj = (1-adj) * self.mask
        v = self.linear_v(x)
        adj_repeat = torch.repeat_interleave(adj.unsqueeze(0), v.size()[0], dim=0)
        att = torch.bmm(adj_repeat, v)
        # att = torch.einsum('ii, jik -> jik', adj, v)
        return att, adj

class MultiGraphInteraction(nn.Module):
    def __init__(self, in_channels, hiden_channels, att_channels, node_list, task_list):
        '''
        in_channels = 12 (Time Steps)
        hiden_channels = [24, 24, 24, 24]  (Hidensize for every dataset)
        att_channels = [36, 24, 68, 21]    (Hidensize in self-att module, which also for every dataset)
        task_list = ['pems03', 'pems04', 'pems07', 'pems08']
        '''
        super(MultiGraphInteraction, self).__init__()
        device_inter = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_list = task_list

        # Message
        self.inputs_lin = nn.ModuleDict({task_list[i]: nn.Linear(in_channels, hiden_channels[i]) for i in range(len(task_list))})
        self.adp_gcn = nn.ModuleDict({task_list[i]: adapt_gcn_operation(node_list[i], hiden_channels[i], hiden_channels[i]*3) for i in range(len(task_list))})
        # self.att_self_graph = nn.ModuleDict({task_list[i]: selfAttention(1, hiden_channels[i], att_channels[i]) for i in range(len(task_list))})   # 论文最优
        # self.att_self_graph = nn.ModuleDict({task_list[i]: AEattention(hiden_channels[i], (hiden_channels[i]+10), att_channels[i]) for i in range(len(task_list))})
        self.att_self_graph = nn.ModuleDict({task_list[i]: Adap_attention(node_list[i], hiden_channels[i], att_channels[i]) for i in range(len(task_list))})
        self.aggregate_lin = nn.ModuleDict({task_list[i]: nn.Linear(hiden_channels[i]*3, att_channels[i]) for i in range(len(task_list))})
        self.rec_T = nn.ModuleDict({task_list[i]: nn.Linear(att_channels[i], in_channels) for i in range(len(task_list))})

        att_act_dict = {}
        for i, task in enumerate(task_list):
            act_task_set = copy.deepcopy(task_list)
            act_task_set.pop(i)
            num_node = node_list[i]
            agg_nodes = copy.deepcopy(node_list)
            agg_nodes.pop(i)
            att_act_dict[task] = [zip(act_task_set, agg_nodes), num_node]
        self.att_act = nn.ModuleDict({task: nn.ModuleDict({act_task: selfAttention(1, agg_nodes, act_task_set[1]) for act_task, agg_nodes in act_task_set[0]}) for task,act_task_set in att_act_dict.items()})

        self.fusion_act = nn.ModuleDict({task_list[i]: nn.Linear(3*node_list[i], node_list[i], bias=False) for i in range(len(task_list))})

        # Aggreate
        self.aggr_GRU = nn.ModuleDict({task_list[i]: Interacte_GRU(node_list[i], int(node_list[i]/2), node_list[i]) for i in range(len(task_list))})
        self.aggr_node_lin = nn.ModuleDict({task_list[i]: nn.Linear(int(node_list[i]/2), node_list[i]) for i in range(len(task_list))})
        self.conv_agg = nn.ModuleDict({task_list[i]: nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1) for i in range(len(task_list))})
        # Update
        self.res_lin = nn.ModuleDict({task_list[i]: nn.Sequential(
                nn.Linear(12, 8),
                nn.Linear(8, 6),
                nn.ReLU(),
                nn.Linear(6, 1)) for i in range(len(task_list))})


    def forward(self, x_list, adj_list):
        return self.propagate(x_list, adj_list)

    def propagate(self, x_list, adj_list):
        M, C = self.message(x_list, adj_list)
        out = self.aggregate(M, C)
        out = self.update(out, x_list)
        return out

    def message(self, x_list, adj_list):
        task_set = list(x_list.keys())
        aggregations = []
        for task in task_set:

            # 增加了一个新的维度作为特征维度，但是需要15G的现存
            # x_lin = self.inputs_lin[task](x_list[task].unsqueeze(-1))    # [B, N, T] —> [B, N, T, 1] —> [B, N, T, H1]
            # gcn_out = self.adp_gcn[task](x_lin, adj_list[task])     #[B, N, T, H1] —> [B, N, T, 2*H1]
            # self_att_H = self.att_self_graph[task](x_lin.reshape(x_lin.size()[0], -1, x_lin.size()[-1]))  #[B, N*T, H1] —> [B, N*T, Att1]
            # m_1 = self.aggregate_lin[task](gcn_out)
            # m_2 = self_att_H.reshape(m_1.size())        # [B, N, T, Att1]
            # m = m_1 + m_2
            # m = self.rec_T[task](m)                                                     # [B, N, T, Att1] —> [B, N, T, 1]
            # m = m.squeeze(-1).transpose(1, 2)                                           # [B, N, T] —> [B, T, N]
            # aggregations.append(m)

            # 将时间维度作为特征维度
            x_lin = self.inputs_lin[task](x_list[task].transpose(1,2))    # [B, T, N] —> [B, N, T] —> [B, N, H1]
            gcn_out = self.adp_gcn[task](x_lin, adj_list[task])           #[B, N, H1] —> [B, N, 2*H1]
            self_att_H, _ = self.att_self_graph[task](x_lin, adj_list[task])                 #[B, N, H1] —> [B, N, Att1]
            m = self.aggregate_lin[task](gcn_out) + self_att_H
            m = self.rec_T[task](m)                                       # [B, N, Att1] —> [B, N, T]
            m = m.transpose(1, 2)                                         # [B, N, T] —> [B, T, N]
            aggregations.append(m)

        act_tensor_list = []
        for i, task in enumerate(task_set):
            act_task_set = copy.deepcopy(task_set)
            act_task_set.pop(i)
            act_tensors = []
            for j, act_task in enumerate(act_task_set):
                agg_infor = self.att_act[task][act_task](x_list[act_task])              # [B, T, Nj] —> [B, T, Ni]
                act_tensors.append(agg_infor)
            act_tensor = torch.cat(act_tensors, dim=2)                                  # [B, T, 3*Ni]
            act_tensor = self.fusion_act[task](act_tensor)                              # [B, T, Ni]
            act_tensor_list.append(act_tensor)

        for i, task in enumerate(task_set):
            aggregations[i] = F.relu(aggregations[i] + x_list[task])        # [B, T, N]

        return aggregations, act_tensor_list

    def aggregate(self, m, c):
        # aggr_out = self.inte_gru(m, c)
        aggr_list = []
        for i, task in enumerate(self.task_list):
            aggr_out, aggr_hid = self.aggr_GRU[task](m[i], c[i])
            # 只输出最后一个隐藏层的结果
            # aggr_out = aggr_out.unsqueeze(1)
            # res_m = self.conv_agg[task](m[i])
            # aggr_out = F.relu(aggr_out + res_m)
            # aggr_list.append(aggr_out)
            # 输出多个隐藏层的结果
            aggr_hid = self.aggr_node_lin[task](aggr_hid)
            aggr_out = F.relu(aggr_hid + m[i])
            aggr_list.append(aggr_out)

        return aggr_list

    def update(self, aggr_out, x_list):
        update_out_list = []
        for i, task in enumerate(self.task_list):
            # update_out = self.res_lin[task](x_list[task].transpose(1, 2)).transpose(1, 2) + aggr_out[i]
            update_out = self.res_lin[task](aggr_out[i].transpose(1, 2)).transpose(1, 2)    # 这里不用线性层，直接输出结果效果更好

            update_out_list.append(update_out)
        return update_out_list


class adapt_gcn_operation(nn.Module):
    '''
    带临接mask矩阵的GCN
    '''
    def __init__(self, num_nodes, in_dim, out_dim):
        super(adapt_gcn_operation, self).__init__()

        device_gru = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.zeros([num_nodes, num_nodes], device=device_gru, requires_grad=True)
        self.mask = nn.Parameter(mask)
        nn.init.xavier_normal_(self.mask, gain=0.0003)

        self.FC = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, adj):
        adj = adj * self.mask
        degree_matrix = torch.sum(adj, dim=1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag_embed(degree_matrix)
        d_a = torch.matmul(degree_matrix, adj)

        out_1 = torch.einsum('nm, bmc->bnc', d_a, x)  # [B, N*T, in_dim]
        out = torch.relu(self.FC(out_1))                                # [B, N*T, out_dim]
        return out




