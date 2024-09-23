import torch
import torch.nn as nn
from LibMTL.model.MuGIL import MultiGraphInteraction
from create_dataset_pems import pems_dataloader
from trainer_pems import Trainer_pems
from LibMTL.utils import set_random_seed
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.loss import MSELoss
from utils import NYCMetric


def parse_args(parser):
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int, help='training epochs')    # 100
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()



def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    task_name = ['pems03', 'pems04', 'pems07', 'pems08']
    node_list = [358, 307, 883, 170]

    task_dict = {task: {'metrics': ['MAE'],
                        'metrics_fn': NYCMetric(),
                        'loss_fn': MSELoss(),
                        'weight': [0]} for task in task_name}    # weight用于计算新结果和旧结果的提升度时的权重，案例可以看出，回归问题可设置为0，分类问题可设置为1，不对会影响best epoch等输出信息

    # prepare dataloaders
    data_loader = pems_dataloader(tasks=task_name, batchsize=params.bs, seq_len=12, pre_len=1)
    train_dataloaders = data_loader['train']
    val_dataloaders = data_loader['val']
    test_dataloaders = data_loader['test']

    # define encoder and decoders
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()

            # flow parameter #
            self.graph_act = MultiGraphInteraction(in_channels=12,
                                                   hiden_channels=[24, 24, 24, 24],
                                                   att_channels=[36, 24, 68, 21],
                                                   node_list=node_list,
                                                   task_list=['pems03', 'pems04', 'pems07', 'pems08'])

            self.adj = Encoder.get_adj()

        def forward(self, inputs):

            out_list = self.graph_act(inputs, self.adj)
            return out_list

        @staticmethod
        def get_adj():
            task_name = ['pems03', 'pems04', 'pems07', 'pems08']
            adj = train_dataloaders.dataset.adj
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for task in task_name:
                adj[task] = torch.tensor(adj[task], dtype=torch.float).to(device)
            return adj

    decoders = nn.ModuleDict({task_name[i]: nn.Linear(node_list[i], node_list[i]) for i in range(len(task_name))})

    nycModel = Trainer_pems(task_dict=task_dict,
                          weighting=weighting_method.__dict__[params.weighting],
                          architecture=architecture_method.__dict__[params.arch],
                          encoder_class=Encoder,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          **kwargs)

    nycModel.train(train_dataloaders=train_dataloaders,
                      val_dataloaders=val_dataloaders,
                      test_dataloaders=test_dataloaders,
                      epochs=params.epochs)

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    params.multi_input = False

    # set general param #
    params.lr = 0.001
    params.weight_decay = 0.0001
    params.arch = 'MMoE_PEMS'
    params.input_size = {'pems03': [12, 358], 'pems04': [12, 307], 'pems07':[12,883], 'pems08':[12, 170]}
    params.num_experts = [2]

    # set optim scheduler param #
    params.scheduler = 'step'  # step, cos, exp,  None
    params.step_size = 15    # lr update step for epoch  (10)
    params.gamma = 0.5      # multiplication factor of lr

    # set device #
    # set_device(params.gpu_id)

    # set weighting strategies #
    # params.weighting = 'CAGrad'

    # set random seed #
    params.save_path = './results'
    params.seed = 40
    set_random_seed(params.seed)
    main(params)

    # for seed in [10, 20, 30, 40]:
    #     params.seed = seed
    #     set_random_seed(params.seed)
    #     path = './result/' + 'seed=' + str(seed)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     params.save_path = path
    #     main(params)
