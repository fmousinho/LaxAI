import torch.nn as nn
import torchvision.models as models

class SiameseNet(nn.Module):
    """
    A Siamese network that uses a pre-trained ResNet as a backbone
    to generate feature embeddings for player crops.
    """
    def __init__(self, embedding_dim=128):
        super(SiameseNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Use a pre-trained ResNet, but remove its final classification layer
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer to be more suitable for small images if needed
        # For example, smaller kernel and stride
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Get the number of features from the backbone's output
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the final layer with our embedding layer
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)

    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def forward(self, x):
        # The forward pass returns the embedding vector
        embedding = self.backbone(x)
        # L2-normalize the embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward_triplet(self, anchor, positive, negative):
        # Helper function to compute embeddings for a triplet
        emb_anchor = self.forward(anchor)
        emb_positive = self.forward(positive)
        emb_negative = self.forward(negative)
        return emb_anchor, emb_positive, emb_negative