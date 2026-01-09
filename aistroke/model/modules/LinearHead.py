import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(embed_dim, num_classes)

        # 可选：和 CosFace 一样用 Xavier 初始化，方便对比
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, feat, label=None):
        """
        feat: (B, E)
        label: (B,) int64 tensor（为了接口统一，这里不使用）
        """
        logits = self.fc(feat)  # (B, num_classes)
        return logits
