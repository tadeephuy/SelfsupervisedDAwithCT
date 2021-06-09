import torch
from torch import nn

class AdaptationModel(nn.Module):
    def __init__(self, backbone, classifier, n_aux_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux = self.create_aux(self.backbone, n_aux_classes)
    
    def forward(self, x, mode=None):
        x = self.backbone(x)
        cls_logits = self.classifier(x)
        if mode == 'aux':
            aux_logits = self.aux(x)
            return cls_logits,aux_logits
        return cls_logits

    @torch.no_grad()
    def create_aux(self, backbone, n_aux_classes):
        x = torch.randn(1, 128, 128)
        x = backbone(x)
        assert len(x)==2, 'Make sure the backbone flattens the features'

        n_feat = x.shape[1]
        return nn.Sequential(
            nn.Linear(n_feat, 512), nn.BatchNorm1d(512), 
            nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(512, n_aux_classes)
        )
