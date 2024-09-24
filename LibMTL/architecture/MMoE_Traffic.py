import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture
from LibMTL.architecture.MMoE_PEMS import MMoE_PEMS

# class MMoE_PEMS(AbsArchitecture):
#     r"""None MMoE

#     """

#     def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
#         super(MMoE_PEMS, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

#         self.input_size = {task: np.array(self.kwargs['input_size'][task], dtype=int).prod() for task in self.task_name}
#         self.num_experts = self.kwargs['num_experts'][0]
#         self.experts_shared = encoder_class()
#         self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size[task], self.num_experts),
#                                                                 nn.Softmax(dim=-1)) for task in self.task_name})

#     def forward(self, inputs, task_name=None):
#         experts_shared_rep = self.experts_shared(inputs)
#         out = {}
#         for k, task in enumerate(self.task_name):
#             if task_name is not None and task != task_name:
#                 continue
#             out[task] = self.decoders[task](experts_shared_rep[k])
#             out[task] = out[task].transpose(1, 2)
#         return out

#     def get_share_params(self):
#         return self.experts_shared.parameters()

#     def zero_grad_share_params(self):
#         self.experts_shared.zero_grad()



# class MMoE_Traffic(MMoE_PEMS):
#     r"""Multi-gate Mixture-of-Experts (MMoE) for PEMS datasets.
#     """

#     def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
#         super(MMoE_Traffic, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

#         self.input_size = {task: np.array(self.kwargs['input_size'][task], dtype=int).prod() for task in self.task_name}
#         self.num_experts = self.kwargs['num_experts'][0]
#         # self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
#         self.experts_shared = encoder_class()
#         self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size[task], self.num_experts),
#                                                                 nn.Softmax(dim=-1)) for task in self.task_name})

#     def forward(self, inputs, task_name=None):
#         # experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
#         experts_shared_rep = self.experts_shared(inputs)
#         experts_shared_rep = torch.stack([s for s in experts_shared_rep])
#         out = {}
#         for task in self.task_name:
#             if task_name is not None and task != task_name:
#                 continue
#             selector = self.gate_specific[task](torch.flatten(inputs[task], start_dim=1))
#             gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
#             gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
#             out[task] = F.relu(self.decoders[task](gate_rep))
#             out[task] = out[task]
#         return out

#     def get_share_params(self):
#         return self.experts_shared.parameters()

#     def zero_grad_share_params(self):
#         self.experts_shared.zero_grad()

class MMoE_Traffic(MMoE_PEMS):
    r"""Multi-gate Mixture-of-Experts (MMoE) for PEMS datasets.
    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE_Traffic, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

        self.input_size = {task: np.array(self.kwargs['input_size'][task], dtype=int).prod() for task in self.task_name}
        self.num_experts = self.kwargs['num_experts'][0]
        # self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
        self.experts_shared = encoder_class()
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size[task], self.num_experts),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})
        # self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size[task], self.num_experts)) for task in self.task_name})
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(self.input_size[task], self.num_experts), requires_grad=True) for task in self.task_name])

    def forward(self, inputs, task_name=None):
        # experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
        experts_shared_rep = self.experts_shared(inputs)
        experts_shared_rep = torch.stack([s for s in experts_shared_rep])
        out = {}
        for k, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            selector = self.gate_specific[task](torch.flatten(inputs[task], start_dim=1))
            selector_2 = torch.matmul(torch.flatten(inputs[task], start_dim=1), torch.flatten(self.w_gates[k], start_dim=1))
            # gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            # gate_rep = torch.einsum('ji, ij... -> j...', selector, experts_shared_rep)
            gate_rep = torch.einsum('ji, ij... -> j...', selector_2, experts_shared_rep)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            # out[task] = F.relu(self.decoders[task](gate_rep))
            out[task] = F.relu(self.decoders[task](gate_rep))
            out[task] = out[task]
        return out

    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad()

# class MMoE_PEMS(AbsArchitecture):
#     r"""Only graph interact.
#     """
#
#     def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
#         super(MMoE_PEMS, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
#
#         self.num_experts = self.kwargs['num_experts'][0]
#         # self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
#         self.experts_shared = encoder_class()
#
#     def forward(self, inputs, task_name=None):
#         # experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
#         experts_shared_rep = self.experts_shared(inputs)
#         out = {}
#         for i, task in enumerate(self.task_name):
#             if task_name is not None and task != task_name:
#                 continue
#             # out[task] = self.decoders[task](experts_shared_rep[i])
#             out[task] = experts_shared_rep[i]
#         return out
#
#     def get_share_params(self):
#         return self.experts_shared.parameters()
#
#     def zero_grad_share_params(self):
#         self.experts_shared.zero_grad()