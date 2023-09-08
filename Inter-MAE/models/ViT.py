import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, model, feat_dim = 768):
        super(ViT, self).__init__()

        self.vit = model
        self.vit.head = nn.Identity()
        self.inv_head = nn.Sequential(
                            nn.Linear(feat_dim, 512, bias = False),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256, bias = False)
                            ) 
        
    def forward(self, x):
        x = self.vit(x)
        x = self.inv_head(x)
        # x0 = > torch.Size([16, 3, 224, 224])
        # x1 = > torch.Size([16, 768])
        # x2 = > torch.Size([16, 256])
        return x
