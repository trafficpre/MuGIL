import torch, os
from LibMTL.utils import count_parameters
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.utils import set_device
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
import torch.nn as nn
from LibMTL.model.MuGIL import MultiGraphInteraction
# from create_dataset_single_input import nyc_dataloader, get_adj_nyc, nyc_Dataset
from create_dataset_pems import pems_dataloader, pems_Dataset
# from create_dataset_bac import pems_dataloader, get_adj_nyc, pems_Dataset
from LibMTL.loss import MSELoss
from utils import NYCMetric
from LibMTL._record import _PerformanceMeter
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import h5py
import matplotlib.pyplot as plt


def main(save_results_flag, load_model_path, result_file):
    def load_model(weighting, architecture, encoder_class, decoders, task_name, rep_grad, multi_input, device, load_path, **kwargs):
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()

        model = MTLmodel(task_name=task_name,
                              encoder_class=encoder_class,
                              decoders=decoders,
                              rep_grad=rep_grad,
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args']).to(device)

        if load_path is not None:
            if os.path.isdir(load_path):
                load_path = os.path.join(load_path, 'best.pt')
            model.load_state_dict(torch.load(load_path), strict=False)
            print('Load Model from - {}'.format(load_path))
        count_parameters(model)

        return model

    def parse_args(parser):
        parser.add_argument('--bs', default=32, type=int, help='batch size')
        parser.add_argument('--epochs', default=20, type=int, help='training epochs')    # 300
        parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
        return parser.parse_args()


    device = torch.device('cuda:0')

    task_name = ['pems03', 'pems04', 'pems07', 'pems08']
    node_list = [358, 307, 883, 170]
    # node_list = [307, 307, 170, 170]
    task_dict = {task: {'metrics': ['MAE'],
                        'metrics_fn': NYCMetric(),
                        'loss_fn': MSELoss(),
                        'weight': [0]} for task in task_name}

    params = parse_args(LibMTL_args)
    params.multi_input = False

    params.arch = 'MMoE_PEMS'
    input_size = {'pems03': [12, 358], 'pems04': [12, 307], 'pems07':[12,883], 'pems08':[12, 170]}
    params.input_size = input_size    # [seq_len, number_nodes_total]
    params.num_experts = [2]

    data_loader = pems_dataloader(tasks=task_name, batchsize=params.bs, seq_len=12, pre_len=1)
    train_dataloaders = data_loader['train']
    val_dataloaders = data_loader['val']
    test_dataloaders = data_loader['test']

    set_device(params.gpu_id)
    kwargs, optim_param, scheduler_param = prepare_args(params)


    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            # self.graph_act_1 = GraphInteraction(in_channels=12, out_channels=27)   #228
            # # self.graph_act_2 = GraphInteraction(in_channels=526, out_channels=263)
            # self.graph_act_3 = GraphInteraction(in_channels=12, out_channels=27)  #207

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

    nycModel = load_model( weighting=weighting_method.__dict__[params.weighting],
                           architecture=architecture_method.__dict__[params.arch],
                           encoder_class=Encoder,
                           decoders=decoders,
                           task_name=task_name,
                           rep_grad=params.rep_grad,
                           multi_input=params.multi_input,
                           optim_param=optim_param,
                           scheduler_param=scheduler_param,
                           device=torch.device('cuda:0'),
                           load_path=load_model_path,
                           **kwargs)


    def _prepare_dataloaders(dataloaders, task_name, multi_input):
        if not multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def _process_data(loader, device, multi_input, task_name):
        try:
            data, label = next(loader[1])    # loader[1].next()
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])   # loader[1].next()

        if not multi_input:
            for task in task_name:
                data[task] = data[task].to(device, non_blocking=True)
                label[task] = label[task].to(device, non_blocking=True)
        else:
            label = label.to(device, non_blocking=True)
        return data, label

    def process_preds(preds):
        return preds

    def _compute_loss(preds, gts, task_name, multi_input, task_num, device, meter):
        if not multi_input:
            train_losses = torch.zeros(task_num).to(device)
            for tn, task in enumerate(task_name):
                train_losses[tn] = meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = meter.losses[task_name]._update_loss(preds, gts)
        return train_losses

    def evaluation(a, b):
        rmse = math.sqrt(mean_squared_error(a, b))
        mae = mean_absolute_error(a, b)
        # F_norm = la.norm(a - b) / la.norm(a)
        # r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        # var = 1 - (np.var(a - b)) / np.var(a)
        if ((a==0).any()) == False:
            mape = np.mean(np.abs((b-a)/a))*100
        elif ((a==0).all()) == True:
            mape = 0
        else:
            pos_nonzero = np.where(a!=0)
            mape = np.mean(np.abs((b[pos_nonzero] - a[pos_nonzero]) / a[pos_nonzero])) * 100
        return rmse, mae, mape

    def compute_performance(prediction, target, data):  # 计算模型性能
        task_set = norm_dataset.task_set
        flow_norm = data.dataset.flow_norm
        performance = {}
        recovered_data = {}
        for task in task_set:
            max_data = flow_norm[task][0]
            min_data = flow_norm[task][1]
            prediction[task] = pems_Dataset.recover_data(max_data, min_data, prediction[task])
            target[task] = pems_Dataset.recover_data(max_data, min_data, target[task])
            # target[np.where(target==0)] = np.mean(target[np.where(target!=0)])
            rmse, mae, mape = evaluation(target[task].reshape(-1), prediction[task].reshape(-1))  #归一化用   accuracy, r2, var
            performance[task] = [rmse, mae, mape]
            recovered_data[task] = [prediction[task], target[task]]

        return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）

    def compute_signal_performance(prediction, target, data, task):
        prediction = prediction.transpose(0, 2, 1)

        flow_norm = data[task].dataset.flow_norm
        max_data = flow_norm[0]
        min_data = flow_norm[1]

        prediction = pems_Dataset.recover_data(max_data, min_data, prediction)
        target = pems_Dataset.recover_data(max_data, min_data, target)

        rmse, mae, mape = evaluation(target.reshape(-1), prediction.reshape(-1))  #归一化用   accuracy, r2, var
        performance = [rmse, mae, mape]
        recovered_data = [prediction, target]

        return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）


    if save_results_flag == True:
        mode = 'test'
        return_improvement = False
        multi_input = False
        task_num = len(task_dict)
        meter = _PerformanceMeter(task_dict, multi_input)
        norm_dataset = pems_Dataset(task_name, mode='test', seq_len=12, pre_len=1)
        test_loader, test_batch = _prepare_dataloaders(test_dataloaders, task_name, multi_input)

        MAE = {task: [] for task in task_name}
        MAPE = {task: [] for task in task_name}
        RMSE = {task: [] for task in task_name}
        # Target = {task: np.zeros([1, 526]) for task in task_name}  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        # Predict = {task: np.zeros([1, 526]) for task in task_name}  # [N, T, D],T=1 # 预测数据的维度
        Target = {task: np.zeros([1, input_size[task][-1]]) for task in task_name}  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        Predict = {task: np.zeros([1, input_size[task][-1]]) for task in task_name}

        result_file = result_file

        nycModel.eval()
        meter.record_time('begin')
        with torch.no_grad():
            if not multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = _process_data(test_loader, device, multi_input, task_name)
                    test_preds = nycModel(test_inputs)
                    test_preds = process_preds(test_preds)
                    test_losses = _compute_loss(test_preds, test_gts, task_name=task_name, multi_input=False, task_num=task_num, device=device, meter=meter)
                    meter.update(test_preds, test_gts)

                    test_preds_dict = {task: test_preds[task].cpu().detach().numpy() for task in task_name}
                    test_gts_dict = {task: test_gts[task].cpu().detach().numpy() for task in task_name}
                    # performance, data_to_save = compute_performance(test_preds_dict, test_gts_dict, norm_dataset)
                    performance, data_to_save = compute_performance(test_preds_dict, test_gts_dict, test_dataloaders)
                    Predict = {task: np.concatenate([Predict[task], data_to_save[task][0]], axis=0) for task in task_name}
                    Target = {task: np.concatenate([Target[task], data_to_save[task][1]], axis=0) for task in task_name}
                    for task in task_name:
                        RMSE[task].append(performance[task][0])
                        MAE[task].append(performance[task][1])
                        if performance[task][2]!=0:
                            MAPE[task].append(performance[task][2])
            else:
                for tn, task in enumerate(task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = _process_data(test_loader[task], device, multi_input, task_name)
                        test_pred = nycModel(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = process_preds(test_pred)
                        test_loss = _compute_loss(test_pred, test_gt, task_name=task, multi_input=multi_input, task_num=task_num, device=device, meter=meter)
                        meter.update(test_pred, test_gt, task)

                        test_preds_dict = test_pred.cpu().detach().numpy()
                        test_gts_dict = test_gt.cpu().detach().numpy()
                        performance, data_to_save = compute_signal_performance(test_preds_dict, test_gts_dict, test_dataloaders, task)
                        Predict[task] = np.concatenate([Predict[task], data_to_save[0]], axis=0)
                        Target[task] = np.concatenate([Target[task], data_to_save[1]], axis=0)
                        RMSE[task].append(performance[0])
                        MAE[task].append(performance[1])
                        if performance[2]!=0:
                            MAPE[task].append(performance[2])
        meter.record_time('end')
        meter.get_score()
        meter.display(epoch=0, mode=mode)
        improvement = meter.improvement
        meter.reinit()

        for task in task_name:
            print(task + "_Performance:  RMSE {:2.3f}    MAE  {:2.3f}    MAPE  {:2.3f} ".format(np.mean(RMSE[task]),np.mean(MAE[task]), np.mean(MAPE[task])))
            Predict[task] = np.delete(Predict[task], 0, axis=0)  # 将第0行的0删除，因为开始定义的时候用0填充，但是时间是从1开始的
            Target[task] = np.delete(Target[task], 0, axis=0)

        file_obj = h5py.File(result_file, "w")  # 将预测值和目标值保存到文件中，因为要多次可视化看看结果
        for task in task_name:
            file_obj["predict_"+task] = Predict[task]
            file_obj["target_"+task] = Target[task]
            file_obj["RMSE_"+task] = RMSE[task]
            file_obj["MAE_"+task] = MAE[task]
            file_obj["MAPE_"+task] = MAPE[task]
        file_obj.close()


    def visualize_result(h5_file, nodes_id, time_se, visualize_file, task):
        file_obj = h5py.File(h5_file, "r")
        prediction = file_obj["predict_"+task][:]
        target = file_obj["target_"+task][:]
        file_obj.close()

        plot_prediction = prediction[time_se[0]: time_se[1], nodes_id]
        plot_target = target[time_se[0]: time_se[1], nodes_id]

        plt.figure()
        plt.grid(True, linestyle="-.", linewidth=1)
        plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker=" ", color="b")
        plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker=" ", color="r")

        plt.legend(["Ground-truth", "Predictions"], loc="upper right")
        plt.xlabel("Time Steps")
        plt.ylabel("Traffic Volume")
        plt.show(block=True)
        plt.savefig(visualize_file + ".png")

    task_vis = 'pems08'
    visualize_result(h5_file=result_file,
    nodes_id = 143, time_se = [230, 230+288*3],
    visualize_file = "green_node_2", task=task_vis)

if __name__=='__main__':
    save_results_flag = True
    load_model_path = './results/best.pt'
    result_file = './results/best.h5'
    main(save_results_flag, load_model_path, result_file)