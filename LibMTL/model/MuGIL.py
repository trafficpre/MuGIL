import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool, dense_diff_pool, models
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


class Interacte_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Interacte_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        device_gru = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        batch_size = input.size(0)
        step_size = input.size(1)
        h = torch.rand(batch_size, self.hidden_size).cuda()
        lisths = []

        for i in range(step_size):
            x = input[:, i, :]
            c = input_c[:, i, :]
            z = torch.sigmoid((torch.mm(x, self.w_xz) + torch.mm(h, self.w_hz) + torch.mm(c, self.w_cz) + self.b_z))
            r = torch.sigmoid((torch.mm(x, self.w_xr) + torch.mm(h, self.w_hr) + torch.mm(c, self.w_cr) + self.b_r))
            h_tilde = torch.tanh(
                (torch.mm(x, self.w_xh) + torch.mm(r * h, self.w_hh) + torch.mm(c, self.w_ch) + self.b_h))
            h = (1 - z) * h + z * h_tilde
            lisths.append(h)
        hs = torch.stack(lisths, dim=1)
        output = self.outlinear(h)
        output = F.relu(output)
        return output, hs


class Adap_attention(nn.Module):
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
        aggr_list = []
        for i, task in enumerate(self.task_list):
            aggr_out, aggr_hid = self.aggr_GRU[task](m[i], c[i])
            aggr_hid = self.aggr_node_lin[task](aggr_hid)
            aggr_out = F.relu(aggr_hid + m[i])
            aggr_list.append(aggr_out)

        return aggr_list

    def update(self, aggr_out, x_list):
        update_out_list = []
        for i, task in enumerate(self.task_list):
            update_out = self.res_lin[task](aggr_out[i].transpose(1, 2)).transpose(1, 2)
            update_out_list.append(update_out)
        return update_out_list


class adapt_gcn_operation(nn.Module):
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




