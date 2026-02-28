import time
import copy  # 复制模型用于SGD基线
import os  # 处理输出路径
import numpy as np  # 数值处理
import torch  # 张量运算
import torch.nn.functional as F  # KL与余弦
import matplotlib.pyplot as plt  # 绘图
from torch.utils.data import DataLoader  # 数据加载器
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data  # 读取客户端数据
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []  # 记录时间
        self.distance_metric = args.distance_metric  # 距离度量方式

    def _emd_1d(self, vec_a, vec_b, max_samples=2048):  # EMD(1D Wasserstein-1)
        if vec_a.numel() == 0 or vec_b.numel() == 0:  # 空向量
            return 0.0  # 返回0
        if vec_a.numel() > max_samples:  # 限制采样A
            idx_a = torch.randperm(vec_a.numel(), device=vec_a.device)[:max_samples]  # 随机索引A
            vec_a = vec_a[idx_a]  # 子采样A
        if vec_b.numel() > max_samples:  # 限制采样B
            idx_b = torch.randperm(vec_b.numel(), device=vec_b.device)[:max_samples]  # 随机索引B
            vec_b = vec_b[idx_b]  # 子采样B
        p = F.softmax(vec_a.view(-1), dim=0)  # 转成分布P
        q = F.softmax(vec_b.view(-1), dim=0)  # 转成分布Q
        cdf_p = torch.cumsum(p, dim=0)  # P的CDF
        cdf_q = torch.cumsum(q, dim=0)  # Q的CDF
        return torch.sum(torch.abs(cdf_p - cdf_q)).item()  # EMD距离

    def _tensor_distance(self, tensor_a, tensor_b):  # 计算张量距离
        if self.distance_metric == "l2":  # L2距离
            return torch.norm(tensor_a - tensor_b, p=2).item()  # 返回L2
        if self.distance_metric == "l1":  # L1距离
            return torch.norm(tensor_a - tensor_b, p=1).item()  # 返回L1
        if self.distance_metric == "cosine":  # 余弦距离
            flat_a = tensor_a.view(-1)  # 拉平A
            flat_b = tensor_b.view(-1)  # 拉平B
            return (1 - F.cosine_similarity(flat_a, flat_b, dim=0, eps=1e-12)).item()  # 1-相似度
        if self.distance_metric == "kl":  # KL散度
            flat_a = tensor_a.view(-1)  # 拉平A
            flat_b = tensor_b.view(-1)  # 拉平B
            p = F.log_softmax(flat_a, dim=0)  # P的log分布
            q = F.softmax(flat_b, dim=0)  # Q分布
            return F.kl_div(p, q, reduction="batchmean").item()  # KL(P||Q)
        if self.distance_metric == "emd":  # EMD距离
            return self._emd_1d(tensor_a, tensor_b)  # 调用EMD
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")  # 非法度量

    def _compute_model_distance(self, model_a, model_b):  # 计算模型距离
        distances = []  # 距离列表
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):  # 遍历参数
            distances.append(self._tensor_distance(p_a.data, p_b.data))  # 逐层距离
        return float(np.mean(distances)) if len(distances) > 0 else 0.0  # 返回均值

    def _compute_layer_distances(self, model_a, model_b):  # 计算逐层距离
        state_a = model_a.state_dict()  # FedAvg权重
        state_b = model_b.state_dict()  # SGD权重
        layer_names = []  # 层名列表
        layer_distances = []  # 距离列表
        for name, tensor in state_a.items():  # 按名对齐
            if name in state_b:  # 只比较同名层
                layer_names.append(name)  # 保存层名
                layer_distances.append(self._tensor_distance(tensor, state_b[name]))  # 层距离
        return layer_names, layer_distances  # 返回数据

    def _build_sgd_loader(self, sgd_client_id=0):  # 构建SGD数据
        if self.dataset == "MNIST":  # 直接用原始MNIST
            import torchvision  # 按需导入
            import torchvision.transforms as transforms  # MNIST预处理
            from torch.utils.data import ConcatDataset  # 拼接train/test
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # 标准化
            data_root = os.path.join("..", "dataset", "MNIST", "rawdata")  # 原始路径
            trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)  # 训练集
            testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)  # 测试集
            fullset = ConcatDataset([trainset, testset])  # 合并MNIST
            return DataLoader(fullset, batch_size=self.batch_size, drop_last=True, shuffle=True)  # 返回加载器

        sgd_train_data = read_client_data(self.dataset, sgd_client_id, is_train=True, few_shot=self.few_shot)  # 兜底单客户端
        return DataLoader(sgd_train_data, self.batch_size, drop_last=True, shuffle=True)  # 返回加载器

    def train(self):
        sgd_model = copy.deepcopy(self.global_model)  # 初始化SGD基线
        sgd_model.to(self.device)  # 放到设备
        sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=self.learning_rate)  # SGD优化器
        sgd_loss = torch.nn.CrossEntropyLoss()  # 交叉熵
        sgd_loader = self._build_sgd_loader()  # SGD数据
        round_distances = {}  # 客户端距离序列

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            sgd_model.train()  # 训练SGD基线
            for _ in range(self.local_epochs):  # 与本地epoch对齐
                for x, y in sgd_loader:  # 遍历SGD数据
                    if type(x) == type([]):  # 兼容文本输入
                        x[0] = x[0].to(self.device)  # 文本上设备
                    else:
                        x = x.to(self.device)  # 图像上设备
                    y = y.to(self.device)  # 标签上设备
                    output = sgd_model(x)  # 前向传播
                    loss = sgd_loss(output, y)  # 计算损失
                    sgd_optimizer.zero_grad()  # 清梯度
                    loss.backward()  # 反向传播
                    sgd_optimizer.step()  # 参数更新

            for client in self.selected_clients:  # 记录客户端距离
                dist = self._compute_model_distance(client.model, sgd_model)  # 客户端vsSGD
                if client.id not in round_distances:  # 初始化序列
                    round_distances[client.id] = []  # 创建列表
                round_distances[client.id].append(dist)  # 追加距离

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        plot_dir = os.path.join("results", "distance_plots", self.dataset, self.algorithm)  # 图表目录
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)  # 创建目录

        layer_names, layer_distances = self._compute_layer_distances(self.global_model, sgd_model)  # 柱状图数据
        if len(layer_names) > 0:
            plt.figure(figsize=(max(10, len(layer_names) * 0.4), 6))  # 画布尺寸
            plt.bar(range(len(layer_names)), layer_distances)  # 画柱状图
            plt.xticks(range(len(layer_names)), layer_names, rotation=90)  # 层名
            plt.ylabel(f"{self.distance_metric.upper()} Distance")  # y轴标签
            plt.tight_layout()  # 紧凑布局
            plt.savefig(os.path.join(plot_dir, f"fedavg_vs_sgd_layer_distance_{self.distance_metric}.png"))  # 保存柱状图
            plt.close()  # 关闭图像

        if len(round_distances) > 0:
            plt.figure(figsize=(8, 5))  # 画布尺寸
            for cid, distances in round_distances.items():  # 每客户端一条线
                plt.plot(range(len(distances)), distances, label=f"client_{cid}")  # 绘制折线
            plt.xlabel("Round")  # x轴标签
            plt.ylabel(f"Client-SGD {self.distance_metric.upper()} Distance")  # y轴标签
            plt.legend(ncol=2, fontsize=8)  # 图例
            plt.tight_layout()  # 紧凑布局
            plt.savefig(os.path.join(plot_dir, f"fedavg_client_vs_sgd_round_distance_{self.distance_metric}.png"))  # 保存折线图
            plt.close()  # 关闭图像

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
