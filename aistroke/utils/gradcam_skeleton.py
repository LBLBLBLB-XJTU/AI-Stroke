import torch

class UpperBodySkeletonGradCAM:
    """
    Grad-CAM for upper-body H36M skeleton
    - Input: skeleton (B, T, V, 3)
    - Output: joint-level importance (B, V)
    """
    # TODO
    # def __init__(self, model):
    #     self.model = model
    #     self.activations = None   # (B, V, E)
    #     self.gradients = None     # (B, V, E)

    #     # 🔥 hook 在「joint token 还没被聚合」的位置
    #     # ⚠️ 如果你的 SkeletonTokenizer 内部名字不同，只改这一行
    #     target_module = model.skeleton_tokenizer

    #     def forward_hook(module, inp, out):
    #         # out: (B, V, E)
    #         self.activations = out

    #     def backward_hook(module, grad_in, grad_out):
    #         # grad_out[0]: (B, V, E)
    #         self.gradients = grad_out[0]

    #     target_module.register_forward_hook(forward_hook)
    #     target_module.register_full_backward_hook(backward_hook)

    # def __call__(self, inputs, side="left"):
    #     """
    #     Args:
    #         inputs: dict with key "skeleton": (B, T, V, 3)
    #         side: "left" or "right"

    #     Returns:
    #         cam: (B, V) joint importance
    #     """
    #     assert side in ["left", "right"]

    #     self.model.zero_grad()

    #     # 🔥 核心：必须让 joints 参与梯度
    #     inputs["joints"].requires_grad_(True)

    #     # ---- forward ----
    #     (feat_l, feat_r), _ = self.model(inputs)

    #     # ---- 选择解释目标 ----
    #     # 用 feature L2 能量，而不是 logit（避免 CosFace 干扰）
    #     if side == "left":
    #         score = feat_l.norm(p=2, dim=1).sum()
    #     else:
    #         score = feat_r.norm(p=2, dim=1).sum()

    #     # ---- backward ----
    #     score.backward(retain_graph=True)

    #     # ===== 正确的 Skeleton Grad-CAM =====
    #     # activations, gradients: (B, V, E)

    #     if self.activations is None or self.gradients is None:
    #         raise RuntimeError("GradCAM hooks did not capture activations / gradients")

    #     # 🔥 不要 mean，不要 norm，直接 Grad × Act
    #     cam = (self.gradients * self.activations).sum(dim=-1)  # (B, V)

    #     # ReLU
    #     cam = torch.relu(cam)

    #     # Normalize per-sample
    #     cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-6)

    #     print(self.gradients.abs().sum(dim=(0,2)))

    #     return cam.detach()
