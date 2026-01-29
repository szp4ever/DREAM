"""
使用Diffusion Model生成Mixup数据

实现平衡采样和跨类别mixup策略
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import json
import random
from collections import defaultdict


from semilearn.datasets import get_cifar, get_food101, get_svhn
from semilearn.imb_algorithms.ours.diffusion_mixup import DiffusionMixup
import torchvision


def get_class_name(dataset_name, label, num_classes):
    """
    根据数据集名称和标签获取类别名称
    
    Args:
        dataset_name: 数据集名称 (cifar10, cifar100, food101, svhn)
        label: 类别标签
        num_classes: 类别数
    
    Returns:
        类别名称字符串
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == 'cifar10':
        # CIFAR-10类别名称
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        if 0 <= label < len(class_names):
            return class_names[label]
    elif dataset_name_lower == 'cifar100':
        # CIFAR-100使用torchvision的类别名称
        try:
            dset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
            if hasattr(dset, 'classes') and label < len(dset.classes):
                return dset.classes[label]
        except:
            # 如果无法加载，使用通用名称
            return f"cifar100_class_{label}"
    elif dataset_name_lower == 'food101':
        # Food101使用torchvision的类别名称
        try:
            dset = torchvision.datasets.Food101(root='./data', split='train', download=False)
            if hasattr(dset, 'classes') and label < len(dset.classes):
                return dset.classes[label]
        except:
            # 如果无法加载，使用通用名称
            return f"food_class_{label}"
    elif dataset_name_lower == 'svhn':
        # SVHN是数字0-9
        if 0 <= label <= 9:
            return str(label)
    
    # 默认情况：返回通用名称
    return f"class_{label}"


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


def get_class_indices(targets):
    """
    获取每个类别的样本索引
    
    Args:
        targets: 标签数组
    
    Returns:
        class_indices: 字典，key为类别，value为该类别的索引列表
    """
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    return class_indices


def balanced_sample_for_mixup(class_indices, num_classes, mixup_ratio, total_labeled):
    """
    平衡采样策略：每个类别作为源类的数量尽可能相同
    
    Args:
        class_indices: 每个类别的索引字典（相对于有标签数据集的索引，不是原始数据索引）
        num_classes: 类别数
        mixup_ratio: mixup图像数量相对于有标签数据的倍数
        total_labeled: 有标签数据总数
    
    Returns:
        source_samples: 采样结果列表，每个元素为 (source1_idx_in_lb, source1_label)
                        source1_idx_in_lb是有标签数据集内的索引（0到total_labeled-1）
    """
    total_mixup = int(total_labeled * mixup_ratio)
    
    # 计算每个类别应该采样多少个作为source1
    samples_per_class = total_mixup // num_classes
    remainder = total_mixup % num_classes
    
    source_samples = []
    
    for class_id in range(num_classes):
        # 获取该类别应该采样的数量
        num_samples = samples_per_class
        if class_id < remainder:
            num_samples += 1
        
        # 从该类别中采样（class_indices中的索引已经是相对于有标签数据集的）
        class_idx_list = class_indices[class_id]
        if len(class_idx_list) == 0:
            continue
        
        # 如果需要的数量超过类别样本数，则重复采样
        sampled_indices = random.choices(class_idx_list, k=num_samples)
        
        for idx in sampled_indices:
            source_samples.append((idx, class_id))
    
    return source_samples


def generate_mixup_samples(
    lb_data,
    lb_targets,
    mixup_generator,
    mixup_ratio,
    output_dir,
    device='cuda',
    batch_size=4,
    resolution=512,
    strength=0.5,
    num_inference_steps=25,
    dataset_name='cifar10',
    num_classes=10,
):
    """
    生成Mixup样本
    
    Args:
        lb_data: 有标签数据的原始图像数据（numpy array或PIL Image列表）
        lb_targets: 有标签数据的标签数组（相对于lb_data的索引）
        mixup_generator: DiffusionMixup实例
        mixup_ratio: mixup图像数量相对于有标签数据的倍数
        output_dir: 输出目录
        device: 设备
        batch_size: 批次大小（暂时未使用）
        resolution: 图像分辨率
        strength: 增强强度
        num_inference_steps: 推理步数
    
    Returns:
        metadata: 包含mixup信息的元数据字典
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 获取每个类别的索引（相对于lb_data的索引）
    class_indices = get_class_indices(lb_targets)
    num_classes = len(class_indices)
    total_labeled = len(lb_targets)
    
    print(f"\n开始生成Mixup样本...")
    print(f"有标签数据总数: {total_labeled}")
    print(f"Mixup比例: {mixup_ratio}")
    print(f"目标Mixup样本数: {int(total_labeled * mixup_ratio)}")
    print(f"类别数: {num_classes}")
    
    # 平衡采样：每个类别作为source1的数量尽可能相同
    source_samples = balanced_sample_for_mixup(
        class_indices, num_classes, mixup_ratio, total_labeled
    )
    
    print(f"实际采样数量: {len(source_samples)}")
    print(f"每个类别采样数: {len(source_samples) // num_classes} (大约)")
    
    # 为每个source1随机选择source2（跨类别）
    mixup_samples = []
    all_classes = list(range(num_classes))
    
    for sample_idx, (source1_idx, source1_label) in enumerate(tqdm(source_samples, desc="生成Mixup")):
        # 随机选择source2（不能和source1同一类别）
        available_classes = [c for c in all_classes if c != source1_label]
        if len(available_classes) == 0:
            continue
        
        source2_label = random.choice(available_classes)
        source2_indices = class_indices[source2_label]
        if len(source2_indices) == 0:
            continue
        
        source2_idx = random.choice(source2_indices)
        
        # 获取原始图像（从lb_data中获取）
        source1_img = get_raw_image_from_data(lb_data, source1_idx)
        source2_img = get_raw_image_from_data(lb_data, source2_idx)
        
        # 选择一个随机类别作为prompt（不能和source1的类别相同）
        # 使用source2的类别作为prompt，确保prompt类别与输入图像类别不同
        prompt_label = source2_label
        
        # 获取prompt类别名称（与输入图像类别不同）
        prompt_class_name = get_class_name(dataset_name, prompt_label, num_classes)
        
        # 准备metadata（使用随机的、与输入图像不同的类别名称）
        metadata = {
            "name": prompt_class_name,
            "super_class": None
        }
        
        # 生成mixup图像（使用source1图像 + 随机类别prompt）
        # 输入图像是source1（类别=source1_label），但prompt是另一个类别（类别=prompt_label）
        try:
            mixed_images, _ = mixup_generator(
                image=[source1_img],  # 使用source1的图像
                label=source1_label,  # 原始标签（用于记录）
                metadata=metadata,  # prompt使用prompt_label的类别名称
                strength=strength,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
            )
            mixup_img = mixed_images[0]
        except Exception as e:
            print(f"生成mixup失败 (source1_idx={source1_idx}, source2_idx={source2_idx}): {e}")
            continue
        
        # 保存mixup图像
        mixup_filename = f"mixup_{sample_idx:06d}.png"
        mixup_path = os.path.join(images_dir, mixup_filename)
        mixup_img.save(mixup_path)
        
        # 记录元数据（包含两个源图像的完整信息）
        mixup_samples.append({
            'mixup_image': f"images/{mixup_filename}",
            # Source1信息（输入图像）
            'source1_idx': int(source1_idx),  # 在有标签数据集中的索引
            'source1_label': int(source1_label),  # Source1的类别标签
            'source1_class_name': get_class_name(dataset_name, source1_label, num_classes),  # Source1的类别名称
            # Source2信息（用于prompt的类别）
            'source2_idx': int(source2_idx),  # 在有标签数据集中的索引
            'source2_label': int(source2_label),  # Source2的类别标签
            'source2_class_name': get_class_name(dataset_name, source2_label, num_classes),  # Source2的类别名称（用作prompt）
            # Mixup信息
            'mixup_label': int(source1_label),  # mixup使用source1的标签
            'prompt_class_name': prompt_class_name,  # 实际使用的prompt类别名称
        })
    
    # 保存元数据
    metadata = {
        'total_mixup': len(mixup_samples),
        'mixup_ratio': mixup_ratio,
        'resolution': resolution,
        'strength': strength,
        'num_inference_steps': num_inference_steps,
        'samples': mixup_samples
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMixup生成完成！")
    print(f"生成了 {len(mixup_samples)} 个Mixup样本")
    print(f"保存目录: {output_dir}")
    
    return metadata


def get_raw_image_from_data(data, idx):
    """
    从原始数据中获取图像（不经过transform）
    
    Args:
        data: 原始图像数据（numpy array或PIL Image列表/数组）
        idx: 索引
    
    Returns:
        PIL Image
    """
    # 获取原始数据
    img = data[idx]
    
    # 如果是numpy array，转换为PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif not isinstance(img, Image.Image):
        # 如果是其他格式，尝试转换
        img = Image.fromarray(np.array(img))
    
    return img.convert('RGB')


def main():
    parser = argparse.ArgumentParser(description='生成Diffusion Mixup数据')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'food101', 'svhn'],
                       help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据集保存目录')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='类别数')
    
    # Mixup参数
    parser.add_argument('--mixup_ratio', type=float, default=1.0,
                       help='Mixup图像数量相对于有标签数据的倍数')
    parser.add_argument('--output_dir', type=str, default='./mixup_data',
                       help='Mixup数据输出目录')
    parser.add_argument('--resolution', type=int, default=32,
                       help='生成图像分辨率')
    parser.add_argument('--strength', type=float, default=0.8,
                       help='Diffusion增强强度')
    parser.add_argument('--num_inference_steps', type=int, default=25,
                       help='Diffusion推理步数')
    
    # Diffusion模型参数
    parser.add_argument('--lora_path', type=str, default=None,
                       help='LoRA权重路径')
    parser.add_argument('--embed_path', type=str, default=None,
                       help='文本嵌入路径')
    parser.add_argument('--prompt', type=str, default="a photo of a {name}",
                       help='提示词模板')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    # 注意：这里需要传入构建数据集时使用的args对象
    # 为了简化，我们可以创建一个临时的args对象
    args = parser.parse_args()
    
    # 创建临时args用于数据集加载（需要与train.py中的参数一致）
    # 这里假设使用默认的长尾分布参数
    import types
    temp_args = types.SimpleNamespace()
    temp_args.dataset = args.dataset
    temp_args.num_classes = args.num_classes
    temp_args.num_labels = 400  # 默认值
    temp_args.ulb_num_labels = 4600  # 默认值
    temp_args.lb_imb_ratio = 100.0
    temp_args.ulb_imb_ratio = 100.0
    temp_args.lb_imb_type = 'exp'
    temp_args.ulb_imb_type = 'exp'
    temp_args.img_size = 32
    temp_args.crop_ratio = 1.0
    temp_args.noise_ratio = 0.0
    temp_args.noise_type = 'sym'
    temp_args.noise_per_class = False
    temp_args.include_lb_to_ulb = False
    temp_args.num_steps = 5
    temp_args.imb_algorithm = None
    temp_args.seed = args.seed
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载有标签数据集（需要获取原始数据和索引）
    print("加载有标签数据集...")
    
    # 先加载数据集获取lb_idx和原始数据
    # 使用已经导入的 get_cifar, get_food101, get_svhn 函数
    if args.dataset.lower() == 'cifar10':
        data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message = \
            get_cifar(temp_args, 'cifar10', data_dir=args.data_dir, include_lb_to_ulb=False)
    elif args.dataset.lower() == 'cifar100':
        data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message = \
            get_cifar(temp_args, 'cifar100', data_dir=args.data_dir, include_lb_to_ulb=False)
    elif args.dataset.lower() == 'food101':
        data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message = \
            get_food101(temp_args, 'food101', data_dir=args.data_dir, include_lb_to_ulb=False)
    elif args.dataset.lower() == 'svhn':
        data, targets, noised_targets, lb_idx, ulb_idx, lb_dset, ulb_dset, eval_dset, lb_count_message = \
            get_svhn(temp_args, 'svhn', data_dir=args.data_dir, include_lb_to_ulb=False)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 使用lb_idx映射到原始数据的索引
    # lb_idx是有标签数据在原始数据中的索引
    lb_targets = np.array(targets[lb_idx])  # 有标签数据的标签
    lb_data = data[lb_idx]  # 有标签数据的原始图像
    
    print(f"有标签数据总数: {len(lb_dset)}")
    
    # 创建Diffusion Mixup生成器
    print("\n初始化Diffusion Mixup生成器...")
    mixup_generator = DiffusionMixup(
        lora_path=args.lora_path,
        embed_path=args.embed_path,
        prompt=args.prompt,
        device=args.device,
    )
    
    # 生成Mixup样本
    metadata = generate_mixup_samples(
        lb_data=lb_data,
        lb_targets=lb_targets,
        mixup_generator=mixup_generator,
        mixup_ratio=args.mixup_ratio,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=4,
        resolution=args.resolution,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        dataset_name=args.dataset,
        num_classes=args.num_classes,
    )
    
    print("\n生成完成！")


if __name__ == '__main__':
    main()

