import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A simple residual block for MLPs:
      - Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> skip connection -> ReLU
    """
    def __init__(self, hidden_dim, dropout_prob=0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Expecting x: [B, hidden_dim]
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        # Residual skip connection
        out += identity
        out = self.relu(out)

        return out


class ResidualAttentionBlock(nn.Module):
    """
    A simple transformer-style block:
      1) Multi-head self-attention
      2) Skip connection + LayerNorm
      3) Feed-forward (linear -> ReLU -> linear)
      4) Skip connection + LayerNorm
    NOTE: 
      This version re-shapes x to (1, B, hidden_dim) so we effectively have
      sequence length T=1, which does not create interesting "temporal" or 
      "token-wise" attention. It just demonstrates usage. For real usage,
      you'd want x to have shape (T, B, hidden_dim) with T>1.
    """
    def __init__(self, hidden_dim, n_heads=4, dropout_prob=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=n_heads,
                                          dropout=dropout_prob,
                                          batch_first=False)  # expects (T, B, E)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x shape: [B, hidden_dim]
        # we forcibly interpret "B" as "batch size" 
        # and pretend we have a sequence length T=1,
        # so we transpose to (T=1, B, D=hidden_dim)
        x_in = x.unsqueeze(0)  # -> [1, B, hidden_dim]

        # MultiheadAttention
        attn_out, _ = self.attn(x_in, x_in, x_in)  # [1, B, hidden_dim]
        
        # First skip connection + layer norm
        x_in = x_in + self.dropout(attn_out)
        x_in = self.ln1(x_in[0]).unsqueeze(0)   # apply LN along last dim

        # Feed-forward
        ff_out = self.ff(x_in)                 # [1, B, hidden_dim]

        # Second skip connection + layer norm
        x_out = x_in + self.dropout(ff_out)
        x_out = self.ln2(x_out[0])             # [B, hidden_dim]

        return x_out  # [B, hidden_dim]


class ResAttLandmarkDeformationModel(nn.Module):
    def __init__(
        self,
        landmark_dim=204,
        num_emotions=8,
        emotion_embedding_dim=16,
        hidden_dim=256,
        num_res_blocks=2,
        dropout_prob=0.0,
        use_attention=True,        # <-- NEW ARG: if True, add an attention block
        n_heads=4
    ):
        super().__init__()

        # Emotion embedding
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)

        # Input dimension: neutral landmarks + emotion embedding
        input_dim = landmark_dim + emotion_embedding_dim

        # Build up our MLP layers in a list
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
        ]

        # Append residual blocks
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout_prob=dropout_prob))

        # Optionally insert an attention block
        # (If T=1, the effect is minimal, but this is how you'd plug it in.)
        if use_attention:
            layers.append(ResidualAttentionBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout_prob=dropout_prob
            ))

        # Final output layer
        layers.append(nn.Linear(hidden_dim, landmark_dim))

        self.regressor = nn.Sequential(*layers)

    def forward(self, neutral_landmarks, emotion_labels, delta=1):
        """
        neutral_landmarks: Tensor of shape [B, landmark_dim]
        emotion_labels:    Tensor of shape [B] (LongTensor)
        delta:             scalar multiplier for the deformation
        """
        # Get emotion embedding
        emotion_embed = self.emotion_embedding(emotion_labels) 
        if emotion_embed.dim() == 1:
            emotion_embed = emotion_embed.unsqueeze(0)  # [1, embedding_dim]
        if emotion_embed.dim() == 3:
            emotion_embed = emotion_embed.squeeze(1)    # [B, embedding_dim]

        # If the batch sizes mismatch for some reason, repeat embed
        if emotion_embed.shape[0] != neutral_landmarks.shape[0]:
            emotion_embed = emotion_embed.repeat(neutral_landmarks.shape[0], 1)

        # Concatenate neutral landmarks with emotion embeddings
        x = torch.cat([neutral_landmarks, emotion_embed], dim=1)
        # x -> [B, input_dim]

        # Regress delta landmarks
        delta_landmarks = self.regressor(x)  # [B, landmark_dim]

        # Compute final emotional landmarks
        emotional_landmarks = neutral_landmarks + delta_landmarks * delta
        return emotional_landmarks


