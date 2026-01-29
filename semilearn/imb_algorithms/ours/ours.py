# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.utils import get_data_loader
from semilearn.datasets.augmentation import RandAugment
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from torch.utils.data import Dataset
from PIL import Image
import json
from .shortcut import GradientShortcutFilter


def compute_semantic_distance(logits):
    semantic = F.softmax(logits, dim=-1)
    return semantic

def compute_mixed_cross_entropy(cls_mix, y_a, y_b, lam_a, lam_b):

    # 计算两个交叉熵损失
    ce_a = F.cross_entropy(cls_mix, y_a, reduction='none')  # [batch_size]
    ce_b = F.cross_entropy(cls_mix, y_b, reduction='none')  # [batch_size]   
    # 使用动态lambda加权
    loss = (lam_a * ce_a + lam_b * ce_b).mean()
    
    return loss

def compute_dynamic_lambda(feats_one, feats_two, feats_mix, y_a, y_b, lam_base=0.5, use_class_specific=False, debug=False):

    # 1. 特征归一化（L2归一化，使距离计算更稳定）
    # 确保特征维度 > 1
    if feats_one.dim() < 2:
        raise ValueError("特征维度必须大于等于2: [batch_size, feat_dim]")
        
    feats_one_norm = F.normalize(feats_one, p=2, dim=-1)  # [batch_size, feat_dim]
    feats_two_norm = F.normalize(feats_two, p=2, dim=-1)  # [batch_size, feat_dim]
    feats_mix_norm = F.normalize(feats_mix, p=2, dim=-1)  # [batch_size, feat_dim]
    
    cos_sim_a = torch.sum(feats_mix_norm * feats_one_norm, dim=-1)  # [batch_size]
    # mixup图像特征与source2特征的相似度
    cos_sim_b = torch.sum(feats_mix_norm * feats_two_norm, dim=-1)  # [batch_size]
    
    alpha = 1.0 - cos_sim_a  # [batch_size]
    alpha_ = 1.0 - cos_sim_b # [batch_size]
    
    T = 1.0  # 原始代码为 0.1，这里改为 1.0，以提高稳定性
    
    # 避免 alpha 或 alpha_ 过大导致 exp(-alpha/T) 下溢
    INa = torch.exp(-alpha / T)  # [batch_size]
    INb = torch.exp(-alpha_ / T) # [batch_size]
    
    # 3. 计算动态Lambda
    # lam_a = (lam_base × INa) / (lam_base × INa + (1-lam_base) × INb)
    
    numerator = lam_base * INa  # [batch_size]
    denominator = lam_base * INa + (1.0 - lam_base) * INb + 1e-8  # 数值稳定性
    
    lam_a = numerator / denominator  # [batch_size]
    lam_b = 1.0 - lam_a  # [batch_size]
    
    # 4. 调试信息
    if debug:
        print(f"\n[Dynamic Lambda Debug - 基于余弦距离 T={T}]")
        print(f"  余弦相似度 A: mean={cos_sim_a.mean():.4f}, range=[{cos_sim_a.min():.4f}, {cos_sim_a.max():.4f}]")
        print(f"  余弦相似度 B: mean={cos_sim_b.mean():.4f}, range=[{cos_sim_b.min():.4f}, {cos_sim_b.max():.4f}]")
        print(f"  Alpha (余弦距离 A): mean={alpha.mean():.4f}, std={alpha.std():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
        print(f"  Alpha_(余弦距离 B): mean={alpha_.mean():.4f}, std={alpha_.std():.4f}, range=[{alpha_.min():.4f}, {alpha_.max():.4f}]")
        print(f"  INa权重: mean={INa.mean():.4f}, range=[{INa.min():.4f}, {INa.max():.4f}]")
        print(f"  INb权重: mean={INb.mean():.4f}, range=[{INb.min():.4f}, {INb.max():.4f}]")
        print(f"  Lambda_a: mean={lam_a.mean():.4f}, std={lam_a.std():.4f}, range=[{lam_a.min():.4f}, {lam_a.max():.4f}]")
        print(f"  Lambda_b: mean={lam_b.mean():.4f}, std={lam_b.std():.4f}, range=[{lam_b.min():.4f}, {lam_b.max():.4f}]")
    
    return lam_a, lam_b, alpha, alpha_

class MixupWithSourcesDataset(Dataset):
    
    def __init__(self, mixup_dataset, lb_dset, weak_transform=None):
        """
        Args:
            mixup_dataset: MixupDataset实例
            lb_dset: 有标签数据集（用于获取原图）
            weak_transform: 弱增强变换（用于mixup图像，应该与lb_dset的weak_transform一致）
        """
        self.mixup_dataset = mixup_dataset
        self.lb_dset = lb_dset
        self.weak_transform = weak_transform
        
        # 如果没有提供transform，尝试从lb_dset获取
        if self.weak_transform is None and hasattr(lb_dset, 'weak_transform'):
            self.weak_transform = lb_dset.weak_transform
    
    def __len__(self):
        return len(self.mixup_dataset)
    
    def __getitem__(self, idx):
        # 获取mixup样本信息
        mixup_sample = self.mixup_dataset[idx]

        source1_idx = mixup_sample['source1_idx']
        source2_idx = mixup_sample['source2_idx']
        
        # 获取原始PIL图像
        source1_img = self.lb_dset.data[source1_idx]
        source2_img = self.lb_dset.data[source2_idx]
        mixup_img = mixup_sample['mixup_image']
        
        # 转换为PIL Image（如果需要）
        if isinstance(source1_img, np.ndarray):
            source1_img = Image.fromarray(source1_img)
        if isinstance(source2_img, np.ndarray):
            source2_img = Image.fromarray(source2_img)
        
        # 使用相同的随机状态对三张图像应用transform，确保增强一致
        # 获取当前的随机状态
        state = torch.get_rng_state()
        
        # 对三张图像应用相同的transform（相同的随机状态）
        if self.weak_transform is not None:
            # 设置相同的随机种子，确保随机操作一致
            seed = random.randint(0, 2**32 - 1)
            
            # 为三张图像设置相同的随机种子
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            source1_tensor = self.weak_transform(source1_img)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            source2_tensor = self.weak_transform(source2_img)
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            mixup_img_tensor = self.weak_transform(mixup_img)
        else:
            # 如果没有transform，使用默认的ToTensor和Normalize
            transform_default = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            source1_tensor = transform_default(source1_img)
            source2_tensor = transform_default(source2_img)
            mixup_img_tensor = transform_default(mixup_img)
        
        # 恢复随机状态（避免影响其他随机操作）
        torch.set_rng_state(state)
        
        return {
            'mixup_image': mixup_img_tensor,  # 应用transform后的tensor
            'source1_image': source1_tensor,  # 使用相同transform的tensor
            'source2_image': source2_tensor,  # 使用相同transform的tensor
            'source1_label': mixup_sample['source1_label'],
            'source2_label': mixup_sample['source2_label'],
            'mixup_label': mixup_sample['mixup_label'],
            'source1_idx': mixup_sample['source1_idx'],
            'source2_idx': mixup_sample['source2_idx'],
        }

class MixupDataset(Dataset):
    """存储Mixup图像和元数据的数据集"""
    
    def __init__(self, mixup_dir):
        """
        Args:
            mixup_dir: 存储mixup数据的目录
        """
        self.mixup_dir = mixup_dir
        self.metadata_file = os.path.join(mixup_dir, 'metadata.json')
        
        # 加载元数据
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.mixup_samples = self.metadata['samples']
    
    def __len__(self):
        return len(self.mixup_samples)
    
    def __getitem__(self, idx):
        sample = self.mixup_samples[idx]
        
        # 加载mixup图像
        mixup_path = os.path.join(self.mixup_dir, sample['mixup_image'])
        mixup_img = Image.open(mixup_path).convert('RGB')
        
        return {
            'mixup_image': mixup_img,
            'source1_idx': sample['source1_idx'],
            'source2_idx': sample['source2_idx'],
            'source1_label': sample['source1_label'],
            'source2_label': sample['source2_label'],
            'mixup_label': sample['mixup_label'],  # 通常是source1的标签
            'mixup_path': mixup_path
        }
        
@IMB_ALGORITHMS.register('ours')
class Ours(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(Ours, self).__init__(args, net_builder, tb_log, logger)

        # Mixup directory
        self.mixup_dir = args.mixup_dir
        if not os.path.exists(self.mixup_dir):
            raise FileNotFoundError(f"Mixup directory not found: {self.mixup_dir}")

        self.shortcut_filter = GradientShortcutFilter(threshold=0.3, top_k_ratio=0.1, debug=False)

        #warm_up epoch
        self.warm_up = args.warm_up

        # dataset update step
        if args.dataset == 'cifar10':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'cifar100':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'food101':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'svhn':
            self.update_step = 5
            self.memory_step = 5

        # augment r and dim
        self.sim_num = None
        self.feat_aug_r = None
        self.fd = self.model.num_features

        # uniform class feature center
        self.optim_cfc = None
        self.cfcd = self.model.num_features

        # adaptive labeled (include the pseudo labeled) data and its dataloader
        self.current_x = None
        self.current_y = None
        self.current_idx = None
        self.current_noise_y = None
        self.current_one_hot_y = None
        self.current_one_hot_noise_y = None

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        self.select_ulb_pseudo_label_distribution = None

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        self.dataset = args.dataset
        self.data = self.dataset_dict['data']
        self.targets = self.dataset_dict['targets']
        self.noised_targets = self.dataset_dict['noised_targets']
        self.lb_idx =  self.dataset_dict['lb_idx']
        self.ulb_idx =  self.dataset_dict['ulb_idx']

        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [0.485, 0.456, 0.406]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
        self.mean['svhn'] = [0.4380, 0.4440, 0.4730]
        self.mean['food101'] = [0.485, 0.456, 0.406]

        self.std['cifar10'] = [0.229, 0.224, 0.225]
        self.std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
        self.std['svhn'] = [0.1751, 0.1771, 0.1744]
        self.std['food101'] = [0.229, 0.224, 0.225]

        if self.dataset == 'food101':
            self.transform_weak = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])
        else:
            self.transform_weak = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

        # compute lb dist
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)

        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).cuda(args.gpu)

        # compute select_ulb and ulb dist
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(args.gpu)

        # compute lb_select_ulb and lb_ulb dist
        lb_ulb_class_dist = lb_class_dist + ulb_class_dist

        self.lb_ulb_dist = torch.from_numpy(lb_ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # ~self.warm_up ce loss only and not select unlabeled data
            if self.epoch < self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # self.warm_up select unlabeled data but still use labeled data only to compute loss
            elif self.epoch == self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

                # only for feature
                self.current_idx = self.lb_idx
                self.current_x = self.data[self.lb_idx]
                self.current_y = self.targets[self.lb_idx]
                self.current_noise_y = self.noised_targets[self.lb_idx]

                self.adaptive_feature_dest = BasicDataset(self.current_idx, self.current_x, self.current_y, self.current_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                self.adaptive_feature_loader = get_data_loader(self.args, self.adaptive_feature_dest, self.args.batch_size, data_sampler=None, num_workers=self.args.num_workers, drop_last=False)

                # reset current labeled data
                self.current_x = None
                self.current_y = None
                self.current_idx = None
                self.current_noise_y = None

                # get the class feature center
                feature_mean_center = torch.zeros(self.num_classes, self.cfcd).cuda(self.args.gpu)
                with torch.no_grad():
                    for data in self.adaptive_feature_loader:
                        if self.args.noise_ratio > 0:
                            y = data['y_lb_noised']
                        else:
                            y = data['y_lb']
                        x = data['x_lb_w']

                        if isinstance(x, dict):
                            x = {k: v.cuda(self.gpu) for k, v in x.items()}
                        else:
                            x = x.cuda(self.gpu)
                        y = y.cuda(self.gpu)

                        features = self.model(x.detach())['feat']

                        for feat, label in zip(features, y):
                            feature_mean_center[label] = feature_mean_center[label] + feat

                feature_mean_center = torch.div(feature_mean_center, self.lb_select_ulb_dist.unsqueeze(1))

                self.optim_cfc = feature_mean_center

                # Create mixup datasets and dataloaderss
                mixup_dataset = MixupDataset(mixup_dir=self.mixup_dir)
                mixup_with_sources_dataset = MixupWithSourcesDataset(mixup_dataset=mixup_dataset,lb_dset=self.dataset_dict['train_lb'])
                self.mixup_loader = get_data_loader(self.args, mixup_with_sources_dataset, self.args.batch_size, data_sampler=self.args.train_sampler, num_iters=self.num_train_iter, num_workers=self.args.num_workers, drop_last=True)


                # generate the sample augment r
                feat_aug_r = torch.zeros(self.num_classes).cuda(self.gpu)
                sim_num = torch.zeros(self.num_classes).cuda(self.gpu)
                with torch.no_grad():
                    for data in self.adaptive_feature_loader:
                        if self.args.noise_ratio > 0:
                            y = data['y_lb_noised']
                        else:
                            y = data['y_lb']
                        x = data['x_lb_w']

                        if isinstance(x, dict):
                            x = {k: v.cuda(self.gpu) for k, v in x.items()}
                        else:
                            x = x.cuda(self.gpu)
                        y = y.cuda(self.gpu)

                        features = self.model(x.detach())['feat']

                        feature_uniform_center_similarity = (torch.sum(torch.mul(features / torch.norm(features, dim=1, keepdim=True), self.optim_cfc[y] / torch.norm(self.optim_cfc[y], dim=1, keepdim=True)), dim=1) + 1.0) / 2.0
                        sim_num[y] += feature_uniform_center_similarity

                self.sim_num = 1 / torch.div(sim_num, self.lb_select_ulb_dist)
                self.feat_aug_r = self.sim_num / torch.sum(self.sim_num)

            else:
                if self.epoch % self.memory_step == 0:
                    if (
                        self.select_ulb_idx is None
                        or self.select_ulb_pseudo_label is None
                        or self.select_ulb_label is None
                        or len(self.select_ulb_idx) == 0
                    ):
                        self.current_idx = self.lb_idx
                        self.current_x = self.data[self.lb_idx]
                        self.current_y = self.targets[self.lb_idx]
                        self.current_noise_y = self.noised_targets[self.lb_idx]
                        current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
                        current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing
                        self.current_one_hot_y = current_one_hot_y
                        self.current_one_hot_noise_y = current_one_hot_noise_y
                        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(self.args.gpu)
                        self.lb_select_ulb_dist = self.lb_dist
                    else:
                        self.current_x = None
                        self.current_y = None
                        self.current_idx = None
                        self.current_noise_y = None
                        self.current_one_hot_y = None
                        self.current_one_hot_noise_y = None
                        self.select_ulb_pseudo_label_distribution = None

                        select_ulb_idx_to_label = {}
                        select_ulb_idx_to_pseudo_label = {}

                        for ulb_idx, ulb_pseudo_label, ulb_label in zip(self.select_ulb_idx, self.select_ulb_pseudo_label, self.select_ulb_label):
                            if ulb_idx.item() in select_ulb_idx_to_label:
                                select_ulb_idx_to_label[ulb_idx.item()].append(ulb_label.item())
                            else:
                                select_ulb_idx_to_label[ulb_idx.item()] = [ulb_label.item()]

                            if ulb_idx.item() in select_ulb_idx_to_pseudo_label:
                                select_ulb_idx_to_pseudo_label[ulb_idx.item()].append(ulb_pseudo_label.item())
                            else:
                                select_ulb_idx_to_pseudo_label[ulb_idx.item()] = [ulb_pseudo_label.item()]

                        select_ulb_unique_idx = torch.unique(self.select_ulb_idx)

                        select_ulb_unique_label = []
                        select_ulb_unique_pseudo_label = []
                        select_ulb_unique_pseudo_label_distribution = []

                        for ulb_unique_idx in select_ulb_unique_idx:
                            ulb_unique_label = select_ulb_idx_to_label[ulb_unique_idx.item()]
                            ulb_unique_pseudo_label = select_ulb_idx_to_pseudo_label[ulb_unique_idx.item()]

                            ulb_unique_pseudo_label_distribution = torch.zeros(self.num_classes)
                            for item in ulb_unique_pseudo_label:
                                ulb_unique_pseudo_label_distribution[item] += 1.0
                            ulb_unique_pseudo_label_distribution = ulb_unique_pseudo_label_distribution / torch.sum(ulb_unique_pseudo_label_distribution)

                            select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))

                            if len(ulb_unique_pseudo_label) > 12:
                                most_common_label = Counter(ulb_unique_pseudo_label).most_common(1)[0][0]
                                most_common_number = Counter(ulb_unique_pseudo_label).most_common(1)[0][1]
                                if most_common_number > 0.8 * len(ulb_unique_pseudo_label):
                                    select_ulb_unique_pseudo_label.append(torch.tensor([most_common_label]))
                                else:
                                    select_ulb_unique_pseudo_label.append(torch.tensor([-1]))
                            else:
                                select_ulb_unique_pseudo_label.append(torch.tensor([-1]))

                            select_ulb_unique_pseudo_label_distribution.append(ulb_unique_pseudo_label_distribution.unsqueeze(0))

                        select_ulb_unique_label = torch.cat(select_ulb_unique_label)
                        select_ulb_unique_pseudo_label = torch.cat(select_ulb_unique_pseudo_label)
                        select_ulb_unique_pseudo_label_distribution = torch.cat(select_ulb_unique_pseudo_label_distribution)

                        self.select_ulb_idx = torch.masked_select(select_ulb_unique_idx.cpu(), select_ulb_unique_pseudo_label != -1)
                        self.select_ulb_label = torch.masked_select(select_ulb_unique_label, select_ulb_unique_pseudo_label != -1)
                        self.select_ulb_pseudo_label = torch.masked_select(select_ulb_unique_pseudo_label, select_ulb_unique_pseudo_label != -1)
                        self.select_ulb_pseudo_label_distribution = select_ulb_unique_pseudo_label_distribution[select_ulb_unique_pseudo_label != -1]

                        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(self.args.gpu)
                        for item in self.select_ulb_pseudo_label:
                            self.select_ulb_dist[int(item)] += 1

                        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

                        self.current_idx = np.concatenate((self.lb_idx, self.select_ulb_idx), axis=0)
                        self.current_x = self.data[self.current_idx]
                        self.current_y = np.concatenate((self.targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                        self.current_noise_y = np.concatenate((self.noised_targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                        current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
                        current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing
                        self.current_one_hot_y = np.concatenate((current_one_hot_y, self.select_ulb_pseudo_label_distribution), axis=0)
                        self.current_one_hot_noise_y = np.concatenate((current_one_hot_noise_y, self.select_ulb_pseudo_label_distribution), axis=0)

                        self.select_ulb_idx = None
                        self.select_ulb_label = None
                        self.select_ulb_pseudo_label = None
                        self.select_ulb_pseudo_label_distribution = None

                if (self.epoch - self.warm_up - self.memory_step) % self.update_step == 0:
                    self.print_fn(str(self.epoch) + ': Update the labeled data.')
                    self.adaptive_lb_dest = BasicDataset(self.current_idx, self.current_x, self.current_one_hot_y, self.current_one_hot_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                    self.adaptive_lb_dest_loader = get_data_loader(self.args, self.adaptive_lb_dest, self.args.batch_size, data_sampler=self.args.train_sampler, num_iters=self.num_train_iter, num_epochs=self.epochs, num_workers=self.args.num_workers, distributed=self.distributed)

                    self.adaptive_feature_dest = BasicDataset(self.current_idx, self.current_x, self.current_y, self.current_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                    self.adaptive_feature_loader = get_data_loader(self.args, self.adaptive_feature_dest, self.args.batch_size, data_sampler=None, num_workers=self.args.num_workers, drop_last=False)

                    self.current_x = None
                    self.current_y = None
                    self.current_idx = None
                    self.current_noise_y = None
                    self.current_one_hot_y = None
                    self.current_one_hot_noise_y = None

                    feature_mean_center = torch.zeros(self.num_classes, self.cfcd).cuda(self.args.gpu)
                    with torch.no_grad():
                        for data in self.adaptive_feature_loader:
                            if self.args.noise_ratio > 0:
                                y = data['y_lb_noised']
                            else:
                                y = data['y_lb']
                            x = data['x_lb_w']

                            if isinstance(x, dict):
                                x = {k: v.cuda(self.gpu) for k, v in x.items()}
                            else:
                                x = x.cuda(self.gpu)
                            y = y.cuda(self.gpu)

                            features = self.model(x.detach())['feat']

                            for feat, label in zip(features, y):
                                feature_mean_center[label] = feature_mean_center[label] + feat

                    feature_mean_center = torch.div(feature_mean_center, self.lb_select_ulb_dist.unsqueeze(1))

                    self.optim_cfc = feature_mean_center

                    feat_aug_r = torch.zeros(self.num_classes).cuda(self.gpu)
                    sim_num = torch.zeros(self.num_classes).cuda(self.gpu)
                    with torch.no_grad():
                        for data in self.adaptive_feature_loader:
                            if self.args.noise_ratio > 0:
                                y = data['y_lb_noised']
                            else:
                                y = data['y_lb']
                            x = data['x_lb_w']

                            if isinstance(x, dict):
                                x = {k: v.cuda(self.gpu) for k, v in x.items()}
                            else:
                                x = x.cuda(self.gpu)
                            y = y.cuda(self.gpu)

                            features = self.model(x.detach())['feat']

                            feature_uniform_center_similarity = (torch.sum(torch.mul(features / torch.norm(features, dim=1, keepdim=True), self.optim_cfc[y] / torch.norm(self.optim_cfc[y], dim=1, keepdim=True)), dim=1) + 1.0) / 2.0
                            sim_num[y] += feature_uniform_center_similarity

                    self.sim_num = 1 / torch.div(sim_num, self.lb_select_ulb_dist)
                    self.feat_aug_r = self.sim_num / torch.sum(self.sim_num)

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            # Create a combined data loader for the training loop
            if self.epoch >= self.warm_up:
                train_loader = zip(self.adaptive_lb_dest_loader, self.loader_dict['train_ulb'], self.mixup_loader)
            else:
                train_loader = zip(self.adaptive_lb_dest_loader, self.loader_dict['train_ulb'])

            for data in train_loader:
                # Unpack data based on the epoch
                if self.epoch >= self.warm_up:
                    data_lb, data_ulb, data_mix = data
                    data_mix = {
                        f"mix_{k}": v.cuda(self.gpu) for k, v in data_mix.items()
                    }
                    if self.it == 0:  # 只在每个epoch的第一次迭代时打印
                        print(f"\n--- Debug data_mix (Epoch: {self.epoch}) ---")
                        for key, value in data_mix.items():
                            print(f"  Key: {key}, Shape: {value.shape}")
                        print("-------------------------------------\n")
                else:
                    data_lb, data_ulb = data
                    data_mix = {}

                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                processed_data = self.process_batch(**data_lb, **data_ulb)
                self.out_dict, self.log_dict = self.train_step(**processed_data, **data_mix)
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb, **kwargs):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s = outputs['logits'][:2 * num_lb].chunk(2)
                aux_logits_x_lb_w, aux_logits_x_lb_s = outputs['aux_logits'][:2 * num_lb].chunk(2)                
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2 * num_lb:].chunk(2)
                aux_logits_x_ulb_w, aux_logits_x_ulb_s = outputs['aux_logits'][2 * num_lb:].chunk(2)                
                feats_x_lb_w, feats_x_lb_s = outputs['feat'][:2 * num_lb].chunk(2)
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2 * num_lb:].chunk(2)
            else:
                outs_x_lb_w = self.model(x_lb_w)
                logits_x_lb_w = outs_x_lb_w['logits']
                aux_logits_x_lb_w = outs_x_lb_w['aux_logits']
                feats_x_lb_w = outs_x_lb_w['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s = outs_x_lb_s['logits']
                aux_logits_x_lb_s = outs_x_lb_s['aux_logits']
                feats_x_lb_s = outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                aux_logits_x_ulb_s = outs_x_ulb_s['aux_logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    aux_logits_x_ulb_w = outs_x_ulb_w['aux_logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb_w': feats_x_lb_w, 'x_lb_s': feats_x_lb_s, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            if self.epoch < self.warm_up:
                # loss for labeled data
                # compute cross entropy loss
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                lb = lb_smooth

                sup_loss = self.ce_loss(logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_pseudo_label_w_smooth = torch.zeros(self.args.uratio * num_lb, self.num_classes).cuda(self.args.gpu)
                aux_pseudo_label_w_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                aux_pseudo_label_w_smooth.scatter_(1, aux_pseudo_label_w.unsqueeze(1), 1.0 - self.args.smoothing)
                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w_smooth, reduction='mean') + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')

                mask = torch.tensor([False]).cuda(self.args.gpu)
                mix_loss = torch.tensor(0.0).cuda(self.args.gpu)
                total_loss = sup_loss + aux_loss
            else:
                # loss for labeled data = self.warm_up / for labeled and pseudo labeled data > self.warm_up
                # augment labeled data
                if self.epoch < self.warm_up + self.memory_step:
                    aug_feats_x_lb_times = (torch.ones_like(lb) * 10.0).int()
                    lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                    lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                    lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                    lb = lb_smooth
                else:
                    if lb.dim() == 2:
                        lb_indices = torch.argmax(lb, dim=1)
                    else:
                        lb_indices = lb.long()
                    aug_feats_x_lb_times = (10.0 * torch.sort(self.lb_select_ulb_dist).values[self.num_classes // 3] / self.lb_select_ulb_dist).int()[lb_indices]

                aug_lbs_x_lb = lb
                aug_feats_x_lb = feats_x_lb_w

                for feat, label, aug_time in zip(feats_x_lb_w[aug_feats_x_lb_times != 0], lb[aug_feats_x_lb_times != 0], aug_feats_x_lb_times[aug_feats_x_lb_times != 0]):
                    repeated_feat = torch.cat([feat.unsqueeze(0) for _ in range(aug_time)], dim=0)
                    repeated_label = torch.cat([label.unsqueeze(0) for _ in range(aug_time)], dim=0)
                    repeated_feat_aug_r = (torch.cat([self.feat_aug_r[torch.argmax(label).item()].unsqueeze(0) for _ in range(aug_time)], dim=0)).view(-1, 1)
                    repeated_feat_norm = torch.cat([F.normalize(feat.unsqueeze(0)) for _ in range(aug_time)], dim=0)

                    aug_lbs_x_lb = torch.cat([aug_lbs_x_lb, repeated_label], dim=0)
                    aug_feats_x_lb = torch.cat([aug_feats_x_lb, repeated_feat + torch.randn(aug_time, self.fd).cuda(self.gpu) * repeated_feat_aug_r * repeated_feat_norm], dim=0)

                aug_logits_x_lb = self.model.classifier(aug_feats_x_lb)

                # compute cross entropy loss for labeled data
                sup_loss = self.ce_loss(aug_logits_x_lb + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), aug_lbs_x_lb, reduction='mean')

                # compute probability
                before_refined_select_ulb_dist = torch.where(self.select_ulb_dist <= min(self.lb_dist), 0, self.select_ulb_dist)

                sorted_select_ulb_dist, _ = torch.sort(torch.unique(before_refined_select_ulb_dist))

                if len(sorted_select_ulb_dist) == 1:
                    refined_select_ulb_dist = torch.ones_like(before_refined_select_ulb_dist)
                else:
                    refined_select_ulb_dist = torch.where(before_refined_select_ulb_dist == 0, sorted_select_ulb_dist[1], before_refined_select_ulb_dist)

                probs_x_ulb_w = self.compute_prob((logits_x_ulb_w + torch.log(refined_select_ulb_dist / torch.sum(refined_select_ulb_dist))).detach())
                probs_x_ulb_s = self.compute_prob((logits_x_ulb_s + torch.log(refined_select_ulb_dist / torch.sum(refined_select_ulb_dist))).detach())

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                targets_lb = lb if lb.dim() == 1 else lb.argmax(dim=1)
                targets_lb = targets_lb.long()
                aux_loss = self.ce_loss(
                    aux_logits_x_ulb_s,
                    aux_pseudo_label_w,
                    reduction='mean'
                ) + self.ce_loss(
                    aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)),
                    targets_lb,
                    reduction='mean'
                )

                # generate unlabeled targets using pseudo label hook
                pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                pseudo_label_s = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_s, use_hard_label=self.use_hard_label, T=self.T, softmax=False)

                mask_w_s = pseudo_label_w == pseudo_label_s

                # calculate mask
                mask_w = probs_x_ulb_w.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))
                mask_s = probs_x_ulb_s.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))

                confidence_mask = torch.logical_and(torch.logical_and(mask_w, mask_s), mask_w_s)
                shortcut_mask = torch.zeros_like(confidence_mask)
                confident_indices = torch.where(confidence_mask)[0]
                if len(confident_indices) > 0:
                    x_ulb_w_confident = x_ulb_w[confidence_mask]
                    x_ulb_s_confident = x_ulb_s[confidence_mask]
                    feats_ulb_w_confident = feats_x_ulb_w[confidence_mask]
                    feats_ulb_s_confident = feats_x_ulb_s[confidence_mask]
                    logits_ulb_w_confident = logits_x_ulb_w[confidence_mask]
                    logits_ulb_s_confident = logits_x_ulb_s[confidence_mask]
                    pseudo_labels_w_confident = pseudo_label_w[confidence_mask]
                    pseudo_labels_s_confident = pseudo_label_s[confidence_mask]
                    shortcut_sub_mask, _, _ = self.shortcut_filter.gradient_shortcut_filter(
                    model=self.model,
                    x_ulb_w=x_ulb_w_confident,
                    x_ulb_s=x_ulb_s_confident,
                    feats_ulb_w=feats_ulb_w_confident,
                    feats_ulb_s=feats_ulb_s_confident,
                    logits_ulb_w=logits_ulb_w_confident,
                    logits_ulb_s=logits_ulb_s_confident,
                    pseudo_labels_w=pseudo_labels_w_confident,
                    pseudo_labels_s=pseudo_labels_s_confident
                    )

    # 6. 将二次筛选通过的样本的索引，映射回原始批次大小的 shortcut_mask
                    if shortcut_sub_mask.any():
                        original_indices_passed_shortcut = confident_indices[shortcut_sub_mask]
                        shortcut_mask[original_indices_passed_shortcut] = True

                # Combine masks (union)
                mask = torch.logical_or(confidence_mask, shortcut_mask)

                # update select_ulb_idx and its pseudo_label
                if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                    self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                    self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                    self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, pseudo_label_w[mask]], dim=0)
                else:
                    self.select_ulb_idx = idx_ulb[mask]
                    self.select_ulb_label = y_ulb[mask]
                    self.select_ulb_pseudo_label = pseudo_label_w[mask]

            

                # 在这里计算您的新损失
                mix_loss = torch.tensor(0.0).cuda(self.args.gpu)
                if 'mix_mixup_image' in kwargs:
                        # Extract mixup data and labels from kwargs
                    mixup_image = kwargs['mix_mixup_image']
                    source1_image = kwargs['mix_source1_image']
                    source2_image = kwargs['mix_source2_image']
                    source1_label = kwargs['mix_source1_label']
                    source2_label = kwargs['mix_source2_label']

                    self.model.eval()
                    with torch.no_grad():
                        outs_mixup = self.model(mixup_image)
                        outs_source1 = self.model(source1_image)
                        outs_source2 = self.model(source2_image)
                    self.model.train()

                        # Calculate dynamic lambda
                    lam_a, lam_b, _, _ = compute_dynamic_lambda(
                            feats_one=outs_source1['feat'],
                            feats_two=outs_source2['feat'],
                            feats_mix=outs_mixup['feat'],
                            y_a=source1_label,
                            y_b=source2_label
                    )

                        # Calculate mixed cross-entropy loss
                    mix_loss = compute_mixed_cross_entropy(
                            cls_mix=outs_mixup['logits'],
                            y_a=source1_label,
                            y_b=source2_label,
                            lam_a=lam_a,
                            lam_b=lam_b
                    )
            total_loss = sup_loss + aux_loss + 0.01*mix_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         aux_loss=aux_loss.item(),
                                         mix_loss=mix_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--warm_up', int, 1),
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--smoothing', float, 0.1),
            SSL_Argument('--mixup_dir', str, r'E:\cvpr\CPG-main\semilearn\imb_algorithms\ours\mixup_data'),
        ]
