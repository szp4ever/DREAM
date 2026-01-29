import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
import random

from generate_mixup import MixupDataset
from shortcut import GradientShortcutFilter
import torch.nn.functional as F
import copy


class EMA:
    """
    Exponential Moving Average (EMA) 模型
    用于生成更稳定的伪标签
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        初始化EMA模型

        Args:
            model: 原始模型
            decay: EMA衰减率（默认0.999，值越大更新越慢）
            device: 设备
        """
        self.decay = decay
        self.device = device

        # 创建EMA模型的副本（深拷贝）
        self.ema_model = copy.deepcopy(model)

        # 将EMA模型设置为eval模式，并禁用梯度计算
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # 将模型移到指定设备
        if device is not None:
            self.ema_model = self.ema_model.to(device)

    def update(self, model):
        """
        使用指数移动平均更新EMA模型参数

        EMA公式: θ_ema = decay * θ_ema + (1 - decay) * θ_model

        Args:
            model: 当前训练模型
        """
        with torch.no_grad():
            # 获取两个模型的参数字典
            ema_params = dict(self.ema_model.named_parameters())
            model_params = dict(model.named_parameters())

            # 更新每个参数
            for name, param in model_params.items():
                if name in ema_params:
                    ema_params[name].data.mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )

            # 更新缓冲区（如BatchNorm的running_mean和running_var）
            ema_buffers = dict(self.ema_model.named_buffers())
            model_buffers = dict(model.named_buffers())

            for name, buffer in model_buffers.items():
                if name in ema_buffers:
                    ema_buffers[name].data.copy_(buffer.data)

    def __call__(self, x):
        """
        使用EMA模型进行前向传播

        Args:
            x: 输入数据

        Returns:
            模型输出
        """
        return self.ema_model(x)

    def eval(self):
        """将EMA模型设置为评估模式"""
        self.ema_model.eval()
        return self

    def train(self):
        """EMA模型始终处于评估模式，此方法仅为兼容性"""
        self.ema_model.eval()
        return self


class MixupWithSourcesDataset(Dataset):

    def __init__(self, mixup_dataset, lb_dset, weak_transform=None):

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

        # 从lb_dset获取source1和source2的原始图像（PIL Image）
        # 注意：需要从data中直接获取，而不是通过__getitem__（因为__getitem__会应用随机transform）
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
            seed = random.randint(0, 2 ** 32 - 1)

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


def compute_semantic_distance(logits):
    """
    计算语义距离

    Args:
        logits: 模型输出的logits [batch_size, num_classes]

    Returns:
        semantic: softmax后的语义向量 [batch_size, num_classes]
    """
    semantic = F.softmax(logits, dim=-1)
    return semantic


def compute_dynamic_lambda(feats_one, feats_two, feats_mix, y_a, y_b, lam_base=0.5, use_class_specific=False,
                           debug=False):
    # 1. 特征归一化（L2归一化，使距离计算更稳定）
    # 确保特征维度 > 1
    if feats_one.dim() < 2:
        raise ValueError("特征维度必须大于等于2: [batch_size, feat_dim]")

    feats_one_norm = F.normalize(feats_one, p=2, dim=-1)  # [batch_size, feat_dim]
    feats_two_norm = F.normalize(feats_two, p=2, dim=-1)  # [batch_size, feat_dim]
    feats_mix_norm = F.normalize(feats_mix, p=2, dim=-1)  # [batch_size, feat_dim]

    # ============= 方案1: 简单版 - 全局特征余弦距离 =============

    # 计算余弦相似度: cos_sim(A, B) = A * B / (|A| * |B|)
    # 由于特征已归一化，cos_sim(A, B) = A * B

    # mixup图像特征与source1特征的相似度
    cos_sim_a = torch.sum(feats_mix_norm * feats_one_norm, dim=-1)  # [batch_size]
    # mixup图像特征与source2特征的相似度
    cos_sim_b = torch.sum(feats_mix_norm * feats_two_norm, dim=-1)  # [batch_size]

    # 转换为余弦距离：Distance = 1 - Cosine Similarity
    # 距离越大，相似度越小
    alpha = 1.0 - cos_sim_a  # [batch_size]
    alpha_ = 1.0 - cos_sim_b  # [batch_size]

    # 2. 计算IN权重（Information Novelty）
    # alpha越大 -> 距离越大 -> IN权重越小

    # 【建议修改点】将温度 T 增大到 1.0 或 5.0，防止 lambda 过于尖锐
    T = 1.0  # 原始代码为 0.1，这里改为 1.0，以提高稳定性

    # 避免 alpha 或 alpha_ 过大导致 exp(-alpha/T) 下溢
    INa = torch.exp(-alpha / T)  # [batch_size]
    INb = torch.exp(-alpha_ / T)  # [batch_size]

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
        print(
            f"  Alpha (余弦距离 A): mean={alpha.mean():.4f}, std={alpha.std():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
        print(
            f"  Alpha_(余弦距离 B): mean={alpha_.mean():.4f}, std={alpha_.std():.4f}, range=[{alpha_.min():.4f}, {alpha_.max():.4f}]")
        print(f"  INa权重: mean={INa.mean():.4f}, range=[{INa.min():.4f}, {INa.max():.4f}]")
        print(f"  INb权重: mean={INb.mean():.4f}, range=[{INb.min():.4f}, {INb.max():.4f}]")
        print(
            f"  Lambda_a: mean={lam_a.mean():.4f}, std={lam_a.std():.4f}, range=[{lam_a.min():.4f}, {lam_a.max():.4f}]")
        print(
            f"  Lambda_b: mean={lam_b.mean():.4f}, std={lam_b.std():.4f}, range=[{lam_b.min():.4f}, {lam_b.max():.4f}]")

    return lam_a, lam_b, alpha, alpha_


def extract_features(model, x):
    """
    从ResNet18模型中提取特征（fc层之前的特征）

    Args:
        model: ResNet18模型
        x: 输入图像 [batch_size, 3, H, W]

    Returns:
        features: 特征向量 [batch_size, 512]
    """
    # ResNet18的特征提取流程
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x


def compute_class_log_adjustment(class_counts, num_classes, device, tau=1.0, normalize=True):
    """
    计算类别分布的log调整项，用于缓解不平衡影响

    Args:
        class_counts: 每个类别的样本数 [num_classes] 或 list
        num_classes: 类别总数
        device: 设备
        tau: 温度参数，控制调整强度（默认1.0）
        normalize: 是否归一化到最大类别频率（默认True）

    Returns:
        log_adjustment: log调整项 [num_classes]，将在logits中减去此值
    """
    if isinstance(class_counts, list):
        class_counts = np.array(class_counts)
    elif isinstance(class_counts, torch.Tensor):
        class_counts = class_counts.cpu().numpy()

    # 计算类别频率（避免除零）
    class_freq = class_counts + 1e-8
    total_samples = class_freq.sum()
    class_freq = class_freq / total_samples

    if normalize:
        # 归一化到最大类别频率（头类为1，尾类为小的正值）
        max_freq = class_freq.max()
        class_freq_normalized = class_freq / max_freq
    else:
        class_freq_normalized = class_freq

    # 计算log调整项：tau * log(class_freq)
    # 头类（频率大）的调整项为正，尾类（频率小）的调整项为负
    # 减去此值会使头类的logits降低，尾类的logits相对提高
    log_adjustment = tau * np.log(class_freq_normalized + 1e-8)

    # 转换为tensor
    log_adjustment = torch.tensor(log_adjustment, dtype=torch.float32, device=device)

    return log_adjustment


def apply_log_adjustment(logits, log_adjustment):
    """
    对logits应用log调整

    Args:
        logits: 原始logits [batch_size, num_classes]
        log_adjustment: log调整项 [num_classes]

    Returns:
        adjusted_logits: 调整后的logits [batch_size, num_classes]
    """
    return logits - log_adjustment.unsqueeze(0)


def compute_mixed_cross_entropy(cls_mix, y_a, y_b, lam_a, lam_b):
    """
    计算混合交叉熵损失

    Loss = lam_a * CE(cls_mix, y_a) + lam_b * CE(cls_mix, y_b)

    Args:
        cls_mix: mixup图像的分类logits [batch_size, num_classes]
        y_a: source1的标签 [batch_size]
        y_b: source2的标签 [batch_size]
        lam_a: 动态lambda_a [batch_size]
        lam_b: 动态lambda_b [batch_size]

    Returns:
        loss: 混合交叉熵损失（标量）
    """
    # 计算两个交叉熵损失
    ce_a = F.cross_entropy(cls_mix, y_a, reduction='none')  # [batch_size]
    ce_b = F.cross_entropy(cls_mix, y_b, reduction='none')  # [batch_size]

    # 使用动态lambda加权
    loss = (lam_a * ce_a + lam_b * ce_b).mean()

    return loss


def train_with_mixup(
        model,
        lb_loader,
        mixup_loader,
        ulb_loader,
        eval_loader,
        args,
        device,
):
    """
    使用Mixup图像进行训练，同时对无标签数据使用梯度短路筛选

    Args:
        model: 模型
        lb_loader: 有标签数据加载器（用于正常的有标签损失）
        mixup_loader: Mixup数据加载器
        ulb_loader: 无标签数据加载器（用于梯度短路筛选和训练）
        eval_loader: 评估数据加载器
        args: 训练参数
        device: 设备
    """
    print("\n" + "=" * 80)
    print("开始使用Mixup图像训练（含梯度短路筛选）")
    print("=" * 80)

    # 计算类别分布的log调整项（用于缓解不平衡影响）
    log_adjustment = None
    if getattr(args, 'use_class_log_adjustment', True):
        # 从args中获取类别分布信息
        ulb_count = getattr(args, 'ulb_count', None)
        lb_count = getattr(args, 'lb_count', None)

        # 调试信息
        print(f"\n[调试] 类别分布信息检查:")
        print(f"  args.ulb_count: {ulb_count} (类型: {type(ulb_count)})")
        print(f"  args.lb_count: {lb_count} (类型: {type(lb_count)})")

        if ulb_count is None:
            # 如果没有ulb_count，尝试使用lb_count
            if lb_count is not None:
                ulb_count = lb_count
                print("使用有标签数据的类别分布作为log调整参考")

        # 检查ulb_count是否为有效列表（非空且包含数字）
        if ulb_count is not None:
            if isinstance(ulb_count, (list, np.ndarray)):
                if len(ulb_count) == 0:
                    print("警告: ulb_count 是空列表，跳过log调整")
                    ulb_count = None
                elif not any(isinstance(x, (int, float, np.integer, np.floating)) and x > 0 for x in ulb_count):
                    print("警告: ulb_count 中没有有效的正数，跳过log调整")
                    ulb_count = None

        if ulb_count is not None:
            tau = getattr(args, 'log_adjustment_tau', 1.0)
            normalize = getattr(args, 'log_adjustment_normalize', True)
            log_adjustment = compute_class_log_adjustment(
                ulb_count, args.num_classes, device, tau=tau, normalize=normalize
            )
            print(f"类别log调整: tau={tau}, normalize={normalize}")
            print(f"  调整范围: [{log_adjustment.min():.4f}, {log_adjustment.max():.4f}]")
            print(f"  调整均值: {log_adjustment.mean():.4f} ± {log_adjustment.std():.4f}")
        else:
            print("警告: 未找到类别分布信息，跳过log调整")
    else:
        print("类别log调整已禁用")

    # 初始化EMA模型（用于生成更稳定的伪标签）
    use_ema = getattr(args, 'use_ema', True)  # 默认使用EMA
    ema_decay = getattr(args, 'ema_decay', 0.999)  # EMA衰减率
    ema_model = None
    if use_ema:
        ema_model = EMA(model, decay=ema_decay, device=device)
        print(f"初始化EMA模型: decay={ema_decay}")
    else:
        print("不使用EMA模型，直接使用训练模型生成伪标签")

    # 初始化梯度短路筛选器（V2版本：使用完整雅可比矩阵）
    # V2版本特点：
    # - 计算完整的雅可比矩阵 J = nabla_F y（所有类别对特征的梯度）
    # - 使用完整的一阶近似估算短路后的logits（更精确）
    # - 计算成本较高（需要循环K次计算梯度），但结果更准确
    shortcut_filter = GradientShortcutFilter(
        threshold=getattr(args, 'shortcut_threshold', 0.1),
        top_k_ratio=getattr(args, 'shortcut_top_k_ratio', 0.3),
        debug=getattr(args, 'shortcut_debug', False)  # 添加debug选项
    )
    print(
        f"梯度短路筛选器 (V2 - 完整雅可比矩阵): threshold={shortcut_filter.threshold}, top_k_ratio={shortcut_filter.top_k_ratio}")

    # 损失函数
    labeled_criterion = nn.CrossEntropyLoss()  # 有标签数据的损失
    ulb_criterion = nn.CrossEntropyLoss(reduction='none')  # 无标签数据的损失（用于伪标签）

    # 优化器
    # 注意：Mixup训练阶段建议使用较小的学习率，因为模型已经在预热阶段学习过
    # 可以使用预热阶段的初始学习率，或者稍微降低（例如0.1倍）
    mixup_lr = getattr(args, 'mixup_lr', None)
    if mixup_lr is None:
        # 如果没有指定，使用初始学习率的0.1倍（更保守，防止破坏预热后的模型）
        mixup_lr = args.lr * 0.1
    print(f"Mixup训练使用学习率: {mixup_lr} (初始学习率: {args.lr})")

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=mixup_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=mixup_lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")

    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.001
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None

    # 训练历史
    history = {
        'train_loss': [],
        'train_loss_labeled': [],
        'train_loss_mixup': [],
        'train_loss_ulb': [],
        'train_acc': [],
        'test_acc': [],
        'ulb_filter_ratio': [],  # 无标签数据筛选比例
        'lambda_a_mean': [],  # Lambda_a 的均值
        'lambda_a_std': [],  # Lambda_a 的标准差
        'lambda_a_min': [],  # Lambda_a 的最小值
        'lambda_a_max': [],  # Lambda_a 的最大值
        'lambda_b_mean': [],  # Lambda_b 的均值
        'lambda_b_std': [],  # Lambda_b 的标准差
        'lambda_b_min': [],  # Lambda_b 的最小值
        'lambda_b_max': [],  # Lambda_b 的最大值
    }

    best_acc = 0.0

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_loss_labeled = 0.0
        train_loss_mixup = 0.0
        train_loss_ulb = 0.0
        n_labeled_batches = 0
        n_mixup_batches = 0
        n_ulb_batches = 0
        total_ulb_samples = 0
        filtered_ulb_samples = 0

        # 收集 lambda 值用于统计
        lambda_a_list = []
        lambda_b_list = []

        # 重置伪标签质量统计（每个epoch重新开始）
        pseudo_label_stats = {
            'total_samples': 0,  # 筛选后的总样本数
            'correct_predictions': 0,  # 筛选后的伪标签正确数
            'per_class_correct': torch.zeros(args.num_classes, dtype=torch.long).to(device),
            'per_class_total': torch.zeros(args.num_classes, dtype=torch.long).to(device),
            'pseudo_labels': [],
            'true_labels': []
        }

        # 创建迭代器
        lb_iter = iter(lb_loader) if lb_loader else None
        mixup_iter = iter(mixup_loader) if mixup_loader else None
        ulb_iter = iter(ulb_loader) if ulb_loader else None

        # 计算总批次数（取所有loader中最大的）
        total_batches = max(
            len(mixup_loader) if mixup_loader else 0,
            len(lb_loader) if lb_loader else 0,
            len(ulb_loader) if ulb_loader else 0
        )

        pbar = tqdm(range(total_batches), desc=f'Epoch {epoch + 1}/{args.epochs}')

        for batch_idx in pbar:
            # 1. 处理有标签数据
            if lb_iter is not None:
                try:
                    lb_batch = next(lb_iter)
                except StopIteration:
                    lb_iter = iter(lb_loader)
                    lb_batch = next(lb_iter)

                x_lb = lb_batch['x_lb_w'].to(device)
                y_lb = lb_batch['y_lb'].to(device)

                # 前向传播
                outputs = model(x_lb)
                if isinstance(outputs, dict):
                    cls_lb = outputs['logits']
                else:
                    cls_lb = outputs

                # 计算有标签损失
                loss_labeled = labeled_criterion(cls_lb, y_lb)
            else:
                loss_labeled = torch.tensor(0.0, device=device)

            # 2. 处理Mixup数据
            if mixup_iter is not None:
                try:
                    mixup_batch = next(mixup_iter)
                except StopIteration:
                    mixup_iter = iter(mixup_loader)
                    mixup_batch = next(mixup_iter)

                # 准备数据
                mixup_img = mixup_batch['mixup_image'].to(device)
                source1_img = mixup_batch['source1_image'].to(device)
                source2_img = mixup_batch['source2_image'].to(device)
                y_a = mixup_batch['source1_label'].to(device)
                y_b = mixup_batch['source2_label'].to(device)

                # 前向传播：同时输入三张图像提取特征并得到分类logits
                # 提取特征（用于计算动态Lambda）

                # 获取分类logits（用于后续损失计算）
                cls_mix_out = model(mixup_img)
                cls_one_out = model(source1_img)
                cls_two_out = model(source2_img)

                if isinstance(cls_mix_out, dict):
                    cls_mix = cls_mix_out['logits']
                    feats_mix = cls_mix_out['feat']
                else:
                    cls_mix = cls_mix_out
                    feats_mix = extract_features(model, mixup_img)

                if isinstance(cls_one_out, dict):
                    cls_one = cls_one_out['logits']
                    feats_one = cls_one_out['feat']
                else:
                    cls_one = cls_one_out
                    feats_one = extract_features(model, source1_img)

                if isinstance(cls_two_out, dict):
                    cls_two = cls_two_out['logits']
                    feats_two = cls_two_out['feat']
                else:
                    cls_two = cls_two_out
                    feats_two = extract_features(model, source2_img)

                # 计算动态Lambda（基于特征）
                lam_a, lam_b, alpha, alpha_ = compute_dynamic_lambda(
                    feats_one, feats_two, feats_mix, y_a, y_b,
                    lam_base=getattr(args, 'lam_base', 0.5),
                    use_class_specific=getattr(args, 'use_class_specific_lambda', False),
                    debug=getattr(args, 'lambda_debug', False)
                )

                # 收集 lambda 值用于统计（转换为 numpy 并收集）
                lambda_a_list.append(lam_a.detach().cpu().numpy())
                lambda_b_list.append(lam_b.detach().cpu().numpy())

                # 计算混合交叉熵损失（作为有标签数据的loss）
                loss_mixup = compute_mixed_cross_entropy(cls_mix, y_a, y_b, lam_a, lam_b)
            else:
                loss_mixup = torch.tensor(0.0, device=device)

            # 3. 处理无标签数据（使用梯度短路筛选）
            loss_ulb = torch.tensor(0.0, device=device)
            if ulb_iter is not None:
                try:
                    ulb_batch = next(ulb_iter)
                except StopIteration:
                    ulb_iter = iter(ulb_loader)
                    ulb_batch = next(ulb_iter)

                x_ulb_w = ulb_batch['x_ulb_w'].to(device)
                x_ulb_s = ulb_batch['x_ulb_s'].to(device)

                # 获取真实标签（如果存在，用于评估伪标签质量）
                y_ulb_true = ulb_batch.get('y_ulb', None)
                if y_ulb_true is not None:
                    y_ulb_true = y_ulb_true.to(device)

                # ========== 步骤1: 在 model.train() 模式下进行一次完整前向传播 ==========
                # 获取用于计算最终损失的 logits 和特征，保留计算图用于反向传播
                # 注意：只进行一次前向传播，避免污染BN统计数据

                # ========== 使用EMA模型生成伪标签 ==========
                # 选择用于生成伪标签的模型（EMA模型或当前训练模型）
                model_for_pseudo = ema_model if ema_model is not None else model

                # 使用选择的模型生成伪标签（使用弱增强）
                with torch.no_grad():
                    model_for_pseudo.eval()
                    # 使用EMA模型或当前模型获取logits（用于生成伪标签）
                    outputs_ulb_w_for_pseudo = model_for_pseudo(x_ulb_w)
                    if isinstance(outputs_ulb_w_for_pseudo, dict):
                        logits_ulb_w_for_pseudo = outputs_ulb_w_for_pseudo['logits']
                    else:
                        logits_ulb_w_for_pseudo = outputs_ulb_w_for_pseudo

                    # 应用类别log调整（缓解不平衡影响）
                    if log_adjustment is not None:
                        logits_ulb_w_for_pseudo = apply_log_adjustment(
                            logits_ulb_w_for_pseudo, log_adjustment
                        )

                    # 生成伪标签（使用弱增强的预测，已应用log调整）
                    pseudo_labels_w = torch.argmax(logits_ulb_w_for_pseudo, dim=-1)
                    pseudo_labels_s = pseudo_labels_w  # 使用相同的伪标签

                # ========== 使用当前训练模型计算损失（保留计算图） ==========
                # 提取特征（保留计算图，用于后续损失计算）
                outputs_ulb_s_main = model(x_ulb_s)
                if isinstance(outputs_ulb_s_main, dict):
                    logits_ulb_s_main_from_feats = outputs_ulb_s_main['logits']
                    feats_ulb_s_main = outputs_ulb_s_main['feat']
                else:
                    feats_ulb_s_main = extract_features(model, x_ulb_s)
                    logits_ulb_s_main_from_feats = model.fc(feats_ulb_s_main)

                # 应用类别log调整（缓解不平衡影响）
                if log_adjustment is not None:
                    logits_ulb_s_main_from_feats = apply_log_adjustment(
                        logits_ulb_s_main_from_feats, log_adjustment
                    )

                # ========== 步骤2: 切换到 model.eval() 模式进行筛选 ==========
                # 保存当前模式状态
                was_training = model.training
                model.eval()  # 冻结BN层，防止BN统计数据被破坏

                try:
                    # 在 eval 模式下，重新计算用于筛选的特征和logits
                    # 注意：筛选时使用当前训练模型（不是EMA模型），因为需要计算梯度
                    # 但伪标签已经使用EMA模型生成，这样可以得到更稳定的伪标签
                    # 这些是分离的（detached）且需要梯度的，用于筛选器的梯度计算
                    with torch.no_grad():
                        # 在eval模式下重新前向传播（用于筛选）
                        outputs_ulb_w_eval = model(x_ulb_w)
                        if isinstance(outputs_ulb_w_eval, dict):
                            logits_ulb_w_eval = outputs_ulb_w_eval['logits']
                        else:
                            logits_ulb_w_eval = outputs_ulb_w_eval

                        outputs_ulb_s_eval = model(x_ulb_s)
                        if isinstance(outputs_ulb_s_eval, dict):
                            logits_ulb_s_eval = outputs_ulb_s_eval['logits']
                        else:
                            logits_ulb_s_eval = outputs_ulb_s_eval

                    # 提取特征（用于筛选，创建独立的计算图）
                    outputs_ulb_w_for_filter = model(x_ulb_w)
                    if isinstance(outputs_ulb_w_for_filter, dict):
                        feats_ulb_w_for_filter = outputs_ulb_w_for_filter['feat'].detach().requires_grad_(True)
                    else:
                        feats_ulb_w_for_filter = extract_features(model, x_ulb_w).detach().requires_grad_(True)

                    outputs_ulb_s_for_filter = model(x_ulb_s)
                    if isinstance(outputs_ulb_s_for_filter, dict):
                        feats_ulb_s_for_filter = outputs_ulb_s_for_filter['feat'].detach().requires_grad_(True)
                    else:
                        feats_ulb_s_for_filter = extract_features(model, x_ulb_s).detach().requires_grad_(True)

                    # 重新计算logits（用于筛选，在独立计算图中）
                    if 'fc' in dir(model):
                        logits_ulb_w_for_filter = model.fc(feats_ulb_w_for_filter)
                        logits_ulb_s_for_filter = model.fc(feats_ulb_s_for_filter)
                    else:
                        logits_ulb_w_for_filter = model.classifier(feats_ulb_w_for_filter)
                        logits_ulb_s_for_filter = model.classifier(feats_ulb_s_for_filter)

                    # 应用类别log调整（筛选时也使用调整后的logits）
                    if log_adjustment is not None:
                        logits_ulb_w_for_filter = apply_log_adjustment(
                            logits_ulb_w_for_filter, log_adjustment
                        )
                        logits_ulb_s_for_filter = apply_log_adjustment(
                            logits_ulb_s_for_filter, log_adjustment
                        )

                    # 使用梯度短路筛选（V2版本：完整雅可比矩阵）
                    # 在eval模式下进行，不影响BN统计数据
                    # V2版本使用完整的雅可比矩阵进行一阶近似，计算更精确但成本较高
                    # 注意：伪标签使用EMA模型生成，但筛选使用当前训练模型计算梯度
                    filter_mask, filtered_pseudo_labels_w, filtered_pseudo_labels_s = shortcut_filter.gradient_shortcut_filter(
                        model=model,  # 使用当前训练模型计算梯度（用于筛选）
                        x_ulb_w=x_ulb_w,
                        x_ulb_s=x_ulb_s,
                        feats_ulb_w=feats_ulb_w_for_filter,
                        feats_ulb_s=feats_ulb_s_for_filter,
                        logits_ulb_w=logits_ulb_w_for_filter,
                        logits_ulb_s=logits_ulb_s_for_filter,
                        pseudo_labels_w=pseudo_labels_w,  # 使用EMA模型生成的伪标签
                        pseudo_labels_s=pseudo_labels_s
                    )
                finally:
                    # ========== 步骤3: 立即切换回 model.train() 模式 ==========
                    if was_training:
                        model.train()

                # 统计筛选结果
                total_ulb_samples += filter_mask.shape[0]
                filtered_count = filter_mask.sum().item()
                filtered_ulb_samples += filtered_count

                # ========== 统计筛选后的伪标签质量（如果真实标签可用） ==========
                if y_ulb_true is not None and filtered_count > 0:
                    # 只统计筛选后的样本（filter_mask为True的样本）
                    filtered_y_ulb_true = y_ulb_true[filter_mask]
                    # 使用筛选后的伪标签（与损失计算中使用的保持一致）
                    # 注意：filtered_pseudo_labels_w 是从 gradient_shortcut_filter 返回的，已经是筛选后的
                    filtered_pseudo_labels_for_stats = filtered_pseudo_labels_w

                    # 计算匹配情况
                    correct_mask = (filtered_pseudo_labels_for_stats == filtered_y_ulb_true)
                    pseudo_label_stats['total_samples'] += filtered_count
                    pseudo_label_stats['correct_predictions'] += correct_mask.sum().item()

                    # 按类别统计
                    for c in range(args.num_classes):
                        class_mask = (filtered_y_ulb_true == c)
                        if class_mask.any():
                            pseudo_label_stats['per_class_total'][c] += class_mask.sum().item()
                            pseudo_label_stats['per_class_correct'][c] += (correct_mask & class_mask).sum().item()

                    # 保存用于后续分析（可选）
                    if getattr(args, 'save_pseudo_label_details', False):
                        pseudo_label_stats['pseudo_labels'].extend(
                            filtered_pseudo_labels_for_stats.cpu().numpy().tolist())
                        pseudo_label_stats['true_labels'].extend(filtered_y_ulb_true.cpu().numpy().tolist())

                # ========== 步骤4: 使用 filter_mask 和 train 模式下得到的 logits 计算损失 ==========
                # 注意：BatchNorm在训练模式下需要batch size > 1，所以至少要2个样本
                if filtered_count > 1:
                    # 使用 filter_mask 筛选伪标签和 logits
                    # 注意：filtered_pseudo_labels_w 是从 gradient_shortcut_filter 返回的，等同于 pseudo_labels_w[filter_mask]
                    # 为了保持一致性，我们使用 filtered_pseudo_labels_w（与统计部分保持一致）
                    filtered_pseudo_labels = filtered_pseudo_labels_w
                    logits_ulb_s_filtered = logits_ulb_s_main_from_feats[filter_mask]  # 使用train模式下保留的计算图

                    # 计算无标签损失（使用伪标签）
                    loss_ulb = ulb_criterion(logits_ulb_s_filtered, filtered_pseudo_labels).mean()
                    n_ulb_batches += 1
                elif filtered_count == 1:
                    # 如果只有1个样本，BatchNorm会报错，跳过该batch
                    loss_ulb = torch.tensor(0.0, device=device)
                else:
                    # 没有样本通过筛选
                    loss_ulb = torch.tensor(0.0, device=device)

            lambda_labeled = getattr(args, 'lambda_labeled', 1.0)
            lambda_mixup = getattr(args, 'lambda_mixup', 1.0)
            lambda_ulb = getattr(args, 'lambda_ulb', 1.0)

            loss = lambda_labeled * loss_labeled + lambda_mixup * loss_mixup + lambda_ulb * loss_ulb

            # 反向传播（确保梯度清理干净）
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸导致模型崩溃）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 更新EMA模型（在每个batch后更新）
            if ema_model is not None:
                ema_model.update(model)

            # 统计
            train_loss += loss.item()
            train_loss_labeled += loss_labeled.item()
            train_loss_mixup += loss_mixup.item()
            train_loss_ulb += loss_ulb.item()

            if lb_iter is not None:
                n_labeled_batches += 1
            if mixup_iter is not None:
                n_mixup_batches += 1

            # 更新进度条
            filter_ratio = filtered_ulb_samples / total_ulb_samples if total_ulb_samples > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss_lb': f'{loss_labeled.item():.4f}',
                'loss_mix': f'{loss_mixup.item():.4f}',
                'loss_ulb': f'{loss_ulb.item():.4f}',
                'filter': f'{filter_ratio:.2%}'
            })

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 计算平均损失
        avg_loss = train_loss / max(n_labeled_batches, n_mixup_batches, n_ulb_batches, 1)
        avg_loss_labeled = train_loss_labeled / n_labeled_batches if n_labeled_batches > 0 else 0.0
        avg_loss_mixup = train_loss_mixup / n_mixup_batches if n_mixup_batches > 0 else 0.0
        avg_loss_ulb = train_loss_ulb / n_ulb_batches if n_ulb_batches > 0 else 0.0
        filter_ratio = filtered_ulb_samples / total_ulb_samples if total_ulb_samples > 0 else 0.0

        # 计算 lambda 统计信息
        if len(lambda_a_list) > 0:
            lambda_a_all = np.concatenate(lambda_a_list)
            lambda_b_all = np.concatenate(lambda_b_list)

            lambda_a_mean = float(np.mean(lambda_a_all))
            lambda_a_std = float(np.std(lambda_a_all))
            lambda_a_min = float(np.min(lambda_a_all))
            lambda_a_max = float(np.max(lambda_a_all))

            lambda_b_mean = float(np.mean(lambda_b_all))
            lambda_b_std = float(np.std(lambda_b_all))
            lambda_b_min = float(np.min(lambda_b_all))
            lambda_b_max = float(np.max(lambda_b_all))
        else:
            lambda_a_mean = lambda_a_std = lambda_a_min = lambda_a_max = 0.0
            lambda_b_mean = lambda_b_std = lambda_b_min = lambda_b_max = 0.0

        # 评估
        from train import evaluate_train_accuracy, evaluate, print_confusion_matrix

        train_acc = evaluate_train_accuracy(model, lb_loader, device) if lb_loader else 0.0
        test_acc, confusion_matrix = evaluate(
            model, eval_loader, device,
            return_confusion_matrix=True,
            num_classes=args.num_classes
        )

        # 记录历史
        history['train_loss'].append(avg_loss)
        history['train_loss_labeled'].append(avg_loss_labeled)
        history['train_loss_mixup'].append(avg_loss_mixup)
        history['train_loss_ulb'].append(avg_loss_ulb)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['ulb_filter_ratio'].append(filter_ratio)
        history['lambda_a_mean'].append(lambda_a_mean)
        history['lambda_a_std'].append(lambda_a_std)
        history['lambda_a_min'].append(lambda_a_min)
        history['lambda_a_max'].append(lambda_a_max)
        history['lambda_b_mean'].append(lambda_b_mean)
        history['lambda_b_std'].append(lambda_b_std)
        history['lambda_b_min'].append(lambda_b_min)
        history['lambda_b_max'].append(lambda_b_max)

        # 计算伪标签准确率（如果统计了）
        pseudo_label_accuracy = 0.0
        if pseudo_label_stats['total_samples'] > 0:
            pseudo_label_accuracy = pseudo_label_stats['correct_predictions'] / pseudo_label_stats[
                'total_samples'] * 100.0

        # 打印结果
        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(
            f"  训练损失: {avg_loss:.4f} (有标签: {avg_loss_labeled:.4f}, Mixup: {avg_loss_mixup:.4f}, 无标签: {avg_loss_ulb:.4f})")
        print(f"  无标签数据筛选比例: {filter_ratio:.2%} ({filtered_ulb_samples}/{total_ulb_samples})")
        if pseudo_label_stats['total_samples'] > 0:
            print(
                f"  筛选后的伪标签准确率: {pseudo_label_accuracy:.2f}% ({pseudo_label_stats['correct_predictions']}/{pseudo_label_stats['total_samples']})")
            # 打印每个类别的准确率
            print(f"  各类别伪标签准确率:")
            for c in range(args.num_classes):
                if pseudo_label_stats['per_class_total'][c].item() > 0:
                    class_acc = pseudo_label_stats['per_class_correct'][c].item() / \
                                pseudo_label_stats['per_class_total'][c].item() * 100.0
                    print(
                        f"    类别 {c}: {class_acc:.2f}% ({pseudo_label_stats['per_class_correct'][c].item()}/{pseudo_label_stats['per_class_total'][c].item()})")
        print(f"  训练集准确率: {train_acc:.2f}%")
        print(f"  验证集准确率: {test_acc:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Lambda统计:")
        print(
            f"    Lambda_a: 均值={lambda_a_mean:.4f}, 标准差={lambda_a_std:.4f}, 范围=[{lambda_a_min:.4f}, {lambda_a_max:.4f}]")
        print(
            f"    Lambda_b: 均值={lambda_b_mean:.4f}, 标准差={lambda_b_std:.4f}, 范围=[{lambda_b_min:.4f}, {lambda_b_max:.4f}]")

        # 打印混淆矩阵

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_model:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc,
                }, os.path.join(args.save_dir, 'mixup_best_model.pth'))
                print(f"  保存最佳模型 (准确率: {best_acc:.2f}%)")

        print("-" * 80)

    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")

    return model, history

