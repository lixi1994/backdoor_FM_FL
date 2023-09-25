import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(30522, d_model, sparse=True)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_mask)
        x = x.mean(dim=1)  # Average pooling
        return self.fc(x)
