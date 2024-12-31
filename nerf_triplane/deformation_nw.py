# audio_deformation_model.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

class AudioFeatureDeformationModel(nn.Module):
    """
    Deforms 'neutral' audio features (like [29,16]) into 'emotional' audio features
    based on an emotion embedding. This parallels a landmark-deformation approach,
    but is for audio-based conditioning features.

    Example usage:
      auds: shape [B, 29, 16] -> flatten to [B, 464] -> pass to model
      emotion_labels: shape [B], each in 0..(num_emotions-1)
    """
    def __init__(
        self,
        audio_feature_dim=512,   
        num_emotions=8,
        emotion_embedding_dim=16,
        hidden_dim=256
    ):
        super(AudioFeatureDeformationModel, self).__init__()
        
        # Embedding for emotion labels (e.g., 8 emotions: angry/happy/etc.)
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)

        # Regressor: input => [audio_feature_dim + emotion_embedding_dim],
        # output => delta of shape [audio_feature_dim].
        input_dim = audio_feature_dim + emotion_embedding_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, audio_feature_dim)
        )

        self.audio_feature_dim = audio_feature_dim

            
    def forward(self, neutral_auds, emotion_labels, delta_scale=1):
# neutral_auds torch.Size([1, 512]), emotion_labels torch.Size([1])
        emotion_embed = self.emotion_embedding(emotion_labels) 
# neutral_auds torch.Size([1, 512]), emotion_embed torch.Size([1, 16])
 
        # print(f'neutral_auds {neutral_auds.shape}, emotion_embed {emotion_embed.shape}')
        # sys.exit()



        x = torch.cat([neutral_auds, emotion_embed], dim=1)  # [171, 528]

        delta_auds = self.regressor(x)
        emotional_auds = neutral_auds + delta_scale * delta_auds

        # emotion_embed = emotion_embed.squeeze(1) # unsqueeze(0) in train, squeeze(1) in test
        # emotion_embed = emotion_embed.repeat(neutral_auds.shape[0], 1)

        return emotional_auds  #171,512

    # def forward(self, neutral_auds, emotion_labels, delta_scale=1.0):
    #     emotion_embed = self.emotion_embedding(emotion_labels) 
    #     emotion_embed_2d = emotion_embed.unsqueeze(0)  # (1,16)

    #     emotion_embed_2d = emotion_embed_2d.expand(64, 16)

    #     neutral_auds = neutral_auds.float() # 64, 1024, 2
    #     neutral_auds = neutral_auds.view(neutral_auds.shape[0], -1) #[64, 2048]
    #     pad_size = 10128 - 2064
    #     neutral_auds = F.pad(neutral_auds, (0, pad_size), mode='constant', value=0)
        # print(f'neutral_auds {neutral_auds.shape}, emotion_embed_2d {emotion_embed_2d.shape}')
        # sys.exit()
    #     # neutral_auds = neutral_auds.cpu().numpy()
    #     # neutral_auds = self.pca.fit_transform(neutral_auds)
    #     # neutral_auds = torch.from_numpy(neutral_auds).to('cuda') # 64, 64
         
    #     # # 3) Unsqueeze again for the 3rd dimension -> (8,16,1)
    #     # emotion_embed_3d = emotion_embed_2d.unsqueeze(-1)

    #     # # 4) Expand to match the 512 dimension -> (8,16,512)
    #     # emotion_embed_3d = emotion_embed_3d.expand(8, 16, 2)


    #     x = torch.cat([neutral_auds, emotion_embed_2d], dim=1)  # [8, 1040]
    #     delta_auds = self.regressor(x)
    #     emotional_auds = neutral_auds + delta_scale * delta_auds
    #     # print(f'x {x.shape}, delta_auds {delta_auds.shape}, emotional_auds {emotional_auds.shape}')
    #     # sys.exit() # x torch.Size([64, 10128]), delta_auds torch.Size([64, 10112]), emotional_auds torch.Size([64, 10112])
    #     return emotional_auds # 64, 10112

#     

 

