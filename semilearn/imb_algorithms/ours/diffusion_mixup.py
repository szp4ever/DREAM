"""
Diffusion Model Mixup实现

使用Stable Diffusion模型进行数据增强，结合LoRA权重和文本嵌入
实现类似于Mixup的数据增强方法
"""

import os
import logging
import torch
from typing import Callable, Tuple, Dict, List, Optional
from PIL import Image
from torch.cuda.amp import autocast


from diffusers import (
        StableDiffusionImg2ImgPipeline,
        DPMSolverMultistepScheduler,
)
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPTextModel, CLIPTokenizer


def format_name(name: str) -> str:
    """
    格式化名称，将空格替换为下划线并添加尖括号
    
    Args:
        name: 原始名称
    
    Returns:
        格式化后的名称
    """
    return f"<{name.replace(' ', '_')}>"


def load_diffmix_embeddings(
    embed_path: str,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device: str = "cuda",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    加载DiffMix的文本嵌入
    
    Args:
        embed_path: 嵌入文件路径
        text_encoder: CLIP文本编码器
        tokenizer: CLIP分词器
        device: 设备
    
    Returns:
        name2placeholder: 名称到占位符的映射
        placeholder2name: 占位符到名称的映射
    """
    ERROR_MESSAGE = "Token {token} already exists in tokenizer"
    
    embedding_ckpt = torch.load(embed_path, map_location="cpu")
    learned_embeds_dict = embedding_ckpt["learned_embeds_dict"]
    name2placeholder = embedding_ckpt["name2placeholder"]
    placeholder2name = embedding_ckpt["placeholder2name"]
    
    # 处理路径分隔符和下划线
    name2placeholder = {
        k.replace("/", " ").replace("_", " "): v 
        for k, v in name2placeholder.items()
    }
    placeholder2name = {
        v: k.replace("/", " ").replace("_", " ") 
        for k, v in name2placeholder.items()
    }
    
    # 添加新token到tokenizer并更新文本编码器
    for token, token_embedding in learned_embeds_dict.items():
        # 添加token到tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)
        
        # 调整token嵌入的大小
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # 获取新添加的token ID
        added_token_id = tokenizer.convert_tokens_to_ids(token)
        
        # 获取旧的词嵌入
        embeddings = text_encoder.get_input_embeddings()
        
        # 将新嵌入分配给新token
        embeddings.weight.data[added_token_id] = token_embedding.to(
            embeddings.weight.dtype
        )
    
    return name2placeholder, placeholder2name


class GenerativeMixup:
    """
    生成式Mixup基类
    """
    def __init__(self):
        pass
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现forward方法")


class DiffusionMixup(GenerativeMixup):
    """
    基于Diffusion模型的Mixup数据增强
    
    使用Stable Diffusion Img2Img管道生成增强图像
    """
    pipe = None  # 全局共享，避免OOM
    
    def __init__(
        self,
        lora_path: Optional[str] = None,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        embed_path: Optional[str] = None,
        prompt: str = "a photo of a {name}",
        format_name: Callable = format_name,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        disable_safety_checker: bool = True,
        revision: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        """
        初始化Diffusion Mixup
        
        Args:
            lora_path: LoRA权重文件路径
            model_path: Stable Diffusion模型路径
            embed_path: 文本嵌入文件路径
            prompt: 提示词模板
            format_name: 名称格式化函数
            guidance_scale: 引导尺度
            disable_safety_checker: 是否禁用安全检查器
            revision: 模型版本
            device: 设备
        """
        super(DiffusionMixup, self).__init__()
        
        # 如果管道尚未初始化，则创建它（全局共享以避免OOM）
        if DiffusionMixup.pipe is None:
            PipelineClass = StableDiffusionImg2ImgPipeline
            
            # 加载Stable Diffusion模型
            DiffusionMixup.pipe = PipelineClass.from_pretrained(
                model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=False,  # 允许从网络下载
                torch_dtype=torch.float16,
            ).to(device)
            
            # 使用DPMSolver调度器
            scheduler = DPMSolverMultistepScheduler.from_config(
                DiffusionMixup.pipe.scheduler.config, 
                local_files_only=False
            )
            DiffusionMixup.pipe.scheduler = scheduler
            
            # 加载文本嵌入（如果有）
            self.placeholder2name = {}
            self.name2placeholder = {}
            
            if embed_path is not None and os.path.exists(embed_path):
                self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(
                    embed_path,
                    DiffusionMixup.pipe.text_encoder,
                    DiffusionMixup.pipe.tokenizer,
                    device=device,
                )
            
            # 加载LoRA权重（如果有）
            if lora_path is not None and os.path.exists(lora_path):
                DiffusionMixup.pipe.load_lora_weights(lora_path)
                print(f"Successfully loaded LoRA weights from {lora_path}!")
            
            # 禁用进度条
            try:
                # 使用diffusers的日志工具
                diffusers_logging.disable_progress_bar()
            except AttributeError:
                # 如果方法不存在，直接使用pipeline的方法
                pass
            DiffusionMixup.pipe.set_progress_bar_config(disable=True)
            
            # 禁用安全检查器（如果需要）
            if disable_safety_checker:
                DiffusionMixup.pipe.safety_checker = None
        
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.format_name = format_name
        self.device = device
    
    def forward(
        self,
        image: List[Image.Image],
        label: int,
        metadata: Dict,
        strength: float = 0.5,
        resolution: int = 512,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[List[Image.Image], int]:
        """
        对图像进行Diffusion增强
        
        使用Diffusion的img2img功能，输入原图和类别prompt，对图像进行修改
        
        Args:
            image: 输入图像列表（PIL Image）- 原图
            label: 标签
            metadata: 元数据字典，应包含'name'字段（类别名称），可选'super_class'字段
            strength: 修改强度（0.0-1.0），值越大变化越大
                      - 0.0: 几乎不改变原图
                      - 1.0: 完全重新生成
            resolution: 生成图像的分辨率
            num_inference_steps: 推理步数（如果为None，使用初始化时的默认值）
                               - 步数越多，生成质量越好但速度越慢
        
        Returns:
            enhanced_images: 增强后的图像列表
            label: 标签（保持不变）
        """
        # 调整图像大小到指定分辨率
        canvas = [
            img.resize((resolution, resolution), Image.BILINEAR) 
            for img in image
        ]
        
        # 获取名称并处理占位符
        name = metadata.get("name", "")
        if self.name2placeholder and name in self.name2placeholder:
            name = self.name2placeholder[name]
        
        # 如果有超类，添加到名称中
        if metadata.get("super_class", None) is not None:
            name = name + " " + metadata.get("super_class", "")
        
        # 格式化提示词
        prompt = self.prompt.format(name=name)
        print(f"Diffusion Mixup prompt: {prompt}")
        
        # 准备生成参数
        # 如果没有指定推理步数，使用初始化时的默认值
        inference_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        
        kwargs = dict(
            image=canvas,  # 原图
            prompt=[prompt] * len(canvas),  # 类别prompt
            strength=strength,  # 修改强度参数
            guidance_scale=self.guidance_scale,
            num_inference_steps=inference_steps,  # 推理步数参数
            num_images_per_prompt=len(canvas),
        )
        
        # 生成增强图像，循环直到没有NSFW内容（如果安全检查器启用）
        has_nsfw_concept = True
        max_retries = 3  # 最大重试次数
        retry_count = 0
        
        while has_nsfw_concept and retry_count < max_retries:
            # 使用autocast加速推理
            # 如果设备是CUDA，使用autocast；否则不使用
            if "cuda" in str(self.device).lower():
                with autocast():
                    outputs = DiffusionMixup.pipe(**kwargs)
            else:
                # CPU设备不使用autocast
                outputs = DiffusionMixup.pipe(**kwargs)
            
            has_nsfw_concept = (
                DiffusionMixup.pipe.safety_checker is not None
                and outputs.nsfw_content_detected
                and any(outputs.nsfw_content_detected)
            )
            
            if has_nsfw_concept:
                retry_count += 1
                print(f"NSFW detected, retrying... ({retry_count}/{max_retries})")
        
        # 将生成的图像调整回原始大小
        enhanced_images = []
        for orig, out in zip(image, outputs.images):
            enhanced_images.append(out.resize(orig.size, Image.BILINEAR))
        
        return enhanced_images, label
    
    def __call__(
        self,
        image: List[Image.Image],
        label: int,
        metadata: Dict,
        strength: float = 0.5,
        resolution: int = 512,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple[List[Image.Image], int]:
        """
        调用forward方法
        """
        return self.forward(image, label, metadata, strength, resolution, num_inference_steps)


def create_diffusion_mixup(
    lora_path: Optional[str] = None,
    embed_path: Optional[str] = None,
    prompt: str = "a photo of a {name}",
    strength: float = 0.5,
    device: str = "cuda",
) -> DiffusionMixup:
    """
    创建Diffusion Mixup增强器的便捷函数
    
    Args:
        lora_path: LoRA权重路径
        embed_path: 文本嵌入路径
        prompt: 提示词模板
        strength: 增强强度
        device: 设备
    
    Returns:
        DiffusionMixup实例
    """
    return DiffusionMixup(
        lora_path=lora_path,
        embed_path=embed_path,
        prompt=prompt,
        device=device,
    )


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="Diffusion Mixup测试")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--embed_path", type=str, default=None, help="文本嵌入路径")
    parser.add_argument("--image_path", type=str, required=True, help="测试图像路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 创建Diffusion Mixup
    mixup = create_diffusion_mixup(
        lora_path=args.lora_path,
        embed_path=args.embed_path,
        device=args.device,
    )
    
    # 加载测试图像
    test_image = Image.open(args.image_path)
    
    # 准备元数据
    metadata = {
        "name": "test_object",
        "super_class": "animal"
    }
    
    # 执行增强
    enhanced_images, label = mixup(
        image=[test_image],
        label=0,
        metadata=metadata,
        strength=0.5,
    )
    
    # 保存结果
    enhanced_images[0].save("enhanced_output.png")
    print("增强完成！结果保存为 enhanced_output.png")

