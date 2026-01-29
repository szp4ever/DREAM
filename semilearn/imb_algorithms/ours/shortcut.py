import torch
import torch.nn.functional as F
class GradientShortcutFilter:

    
    def __init__(self, threshold=0.3, top_k_ratio=0.1, debug=False):
        self.threshold = threshold
        self.top_k_ratio = top_k_ratio
        self.debug = debug
        
        if self.debug:
            print(f"[GradientShortcutFilter V1] 初始化")
            print(f"  - threshold: {threshold}")
            print(f"  - top_k_ratio: {top_k_ratio}")
    
    def compute_energy_score(self, logits):

        return -torch.logsumexp(logits, dim=-1)
    
    def gradient_shortcut_filter(
        self, 
        model, 
        x_ulb_w, 
        x_ulb_s, 
        feats_ulb_w, 
        feats_ulb_s, 
        logits_ulb_w, 
        logits_ulb_s, 
        pseudo_labels_w, 
        pseudo_labels_s
    ):
 
        batch_size = x_ulb_w.shape[0]
        device = x_ulb_w.device
        
        if self.debug:
            print(f"\n{'='*70}")
            print(f"[筛选开始 (完整逻辑 V2)] batch_size={batch_size}")
        
        # 步骤1: 计算原始能量分数
        original_energy = self.compute_energy_score(logits_ulb_w)
        
        # 步骤2: 计算预测类别的logit对特征的梯度 (g)
        # (这只用于寻找 Top-K)
        g_gradients = self._compute_logits_gradients(
            model, feats_ulb_w, logits_ulb_w, pseudo_labels_w
        )
        
        # 步骤3: 识别高梯度幅度的特征坐标
        top_k_indices = self._get_top_k_gradient_indices(g_gradients)
        
        # 步骤4: 计算特征变化量（短路后的特征 delta_F）
        delta_feats = self._compute_feature_delta(feats_ulb_w, top_k_indices)
        
        # 步骤5: (V2) 计算完整的雅可比矩阵 (J)
        # (这是计算成本最高的部分!)
        jacobian = self._compute_logits_jacobian(model, feats_ulb_w)
        
        # 步骤6: (V2) 使用完整的一阶近似估算短路后的logits
        approximated_logits = self._taylor_approximation_full(
            logits_ulb_w, jacobian, delta_feats
        )
        
        # 步骤7: 计算短路后的能量分数
        shortcut_energy = self.compute_energy_score(approximated_logits)
        
        # 步骤8: 计算能量分数变化率
        energy_change_ratio = (shortcut_energy - original_energy) / (torch.abs(original_energy) + 1e-8)
        
        # ... (后续的 debug 和 return 逻辑与 V1 相同) ...
        
        if self.debug:
            self._print_debug_info(
                original_energy, shortcut_energy, energy_change_ratio, batch_size
            )
        
        final_mask = energy_change_ratio <= self.threshold
        
        if self.debug:
            print(f"[筛选结果] 保留 {final_mask.sum().item()}/{batch_size} 个样本 "
                    f"({final_mask.float().mean()*100:.1f}%)")
            print(f"{'='*70}\n")
        
        return (
            final_mask,
            pseudo_labels_w[final_mask],
            pseudo_labels_w[final_mask] 
        )
    
    def _compute_logits_gradients(self, model, feats, logits, pseudo_labels):

        batch_size = feats.shape[0]
        device = feats.device
        
        # 分离特征并设置requires_grad
        feats_copy = feats.detach().clone().requires_grad_(True)
        
        # 重新前向传播获取logits
        if hasattr(model, 'classifier'):
            new_logits = model.classifier(feats_copy)
        elif hasattr(model, 'fc'):
            new_logits = model.fc(feats_copy)
        else:
            raise ValueError(
                "模型必须包含classifier或fc属性用于计算梯度。"
                "请确保你的模型有分类层。"
            )
        
        # 向量化计算：为每个样本选择对应类别的logit
        # 创建一个one-hot向量来选择对应的logit
        batch_indices = torch.arange(batch_size, device=device)
        target_logits = new_logits[batch_indices, pseudo_labels]  # [batch_size]
        
        # 计算所有样本的梯度（一次性计算）
        # outputs是每个样本对应类别的logit，inputs是整个batch的特征
        gradients = torch.autograd.grad(
            outputs=target_logits,
            inputs=feats_copy,
            grad_outputs=torch.ones_like(target_logits),  # 每个输出权重为1
            create_graph=False,  # 不创建计算图，避免影响主训练
            retain_graph=False,  # 不需要保留图
            only_inputs=True
        )[0]
        
        # 返回梯度（已经是[batch_size, feat_dim]形状）
        return gradients.detach()
    
    def _get_top_k_gradient_indices(self, gradients):

        batch_size, feat_dim = gradients.shape
        
        # 计算梯度幅度
        grad_magnitude = torch.abs(gradients)
        
        # 选择top-k比例的特征
        top_k = max(1, int(feat_dim * self.top_k_ratio))
        
        # 获取top-k索引（按梯度幅度排序）
        _, top_k_indices = torch.topk(grad_magnitude, top_k, dim=-1)
        
        if self.debug:
            print(f"[Top-K选择] 每个样本选择 {top_k}/{feat_dim} 个特征进行短路")
        
        return top_k_indices
    
    def _compute_feature_delta(self, feats, top_k_indices):

        batch_size, feat_dim = feats.shape
        device = feats.device
        
        # 创建特征变化量张量
        delta_feats = torch.zeros_like(feats)
        
        # 对每个样本，将top-k特征坐标清零
        for i in range(batch_size):
            delta_feats[i, top_k_indices[i]] = -feats[i, top_k_indices[i]]
        
        return delta_feats
    
    def _compute_logits_jacobian(self, model, feats):

        B, d = feats.shape
        device = feats.device
        
        # 重新计算 logits 以构建计算图
        feats_copy = feats.detach().clone().requires_grad_(True)
        if hasattr(model, 'classifier'):
            new_logits = model.classifier(feats_copy)
        elif hasattr(model, 'fc'):
            new_logits = model.fc(feats_copy)
        else:
            raise ValueError("模型必须包含classifier或fc属性。")
            
        K = new_logits.shape[1] # 获取类别数
        jacobian = torch.zeros(B, K, d, device=device)
        
        # 必须循环 K 次，为每个 logit 计算一次梯度
        for k in range(K):
            # 选择所有 batch 中第 k 个类的 logit
            target_logits_k = new_logits[:, k]
            
            # 如果不是最后一次循环，必须保留计算图
            retain_graph = (k < K - 1)
            
            # 计算第 k 个 logit 对 F 的梯度
            grad_k = torch.autograd.grad(
                outputs=target_logits_k,
                inputs=feats_copy,
                grad_outputs=torch.ones_like(target_logits_k),
                retain_graph=retain_graph,
                create_graph=False,
                only_inputs=True
            )[0]
            
            # 存入雅可比矩阵
            jacobian[:, k, :] = grad_k
            
        return jacobian.detach() 
        
    def _taylor_approximation_full(self, original_logits, jacobian, delta_feats):

        
        # 将 delta_feats 扩展为 [B, d, 1] 以进行批量矩阵乘法
        delta_feats_expanded = delta_feats.unsqueeze(-1)
        
        # 批量矩阵乘法 (BMM)
        # J @ delta_F
        # [B, K, d] @ [B, d, 1] -> [B, K, 1]
        logit_changes = torch.bmm(jacobian, delta_feats_expanded)
        
        # 压缩回 [B, K]
        logit_changes = logit_changes.squeeze(-1)
        
        # y' = y + delta_y
        approximated_logits = original_logits + logit_changes
        
        return approximated_logits
    
    def _taylor_approximation(self, original_logits, gradients, delta_feats):

        # 计算预测类别的logit变化量
        # gradient: [batch_size, feat_dim]
        # delta_feats: [batch_size, feat_dim]
        # 点积: [batch_size]
        logit_change = torch.sum(gradients * delta_feats, dim=-1, keepdim=True)
        
        # 将变化量应用到所有类别（简化假设：所有类别受相似影响）
        # 注意：这里虽然还是加相同的值，但gradients本身已经是正确的logit梯度
        # 而不是原来的损失梯度，所以结果是有意义的
        approximated_logits = original_logits + logit_change
        
        return approximated_logits
    
    def _print_debug_info(self, original_energy, shortcut_energy, energy_change_ratio, batch_size):
        """打印调试信息"""
        print(f"\n[能量分数统计]")
        print(f"  原始能量:")
        print(f"    - 范围: [{original_energy.min():.4f}, {original_energy.max():.4f}]")
        print(f"    - 均值: {original_energy.mean():.4f} ± {original_energy.std():.4f}")
        print(f"  短路后能量:")
        print(f"    - 范围: [{shortcut_energy.min():.4f}, {shortcut_energy.max():.4f}]")
        print(f"    - 均值: {shortcut_energy.mean():.4f} ± {shortcut_energy.std():.4f}")
        print(f"  能量变化率:")
        print(f"    - 范围: [{energy_change_ratio.min():.4f}, {energy_change_ratio.max():.4f}]")
        print(f"    - 均值: {energy_change_ratio.mean():.4f} ± {energy_change_ratio.std():.4f}")
        print(f"    - 阈值: {self.threshold} (变化率 <= 阈值时保留)")
        print(f"    - 变化率 <= 阈值的样本数: {(energy_change_ratio <= self.threshold).sum().item()}/{batch_size}")