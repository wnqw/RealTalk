import sys
import torch
import torch.nn as nn

class AudioDeformationModel(nn.Module):
    def __init__(
        self,
        audio_feature_dim=464,   
        num_emotions=8,
        emotion_embedding_dim=16,
        hidden_dim=256
    ):
        super(AudioDeformationModel, self).__init__()
        
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
# neutral_auds torch.Size([1, 512]), emotion_labels torch.Size([1]) train
# neutral_auds torch.Size([171, 512]), emotion_labels torch.Size([1]) test
        emotion_embed = self.emotion_embedding(emotion_labels) 
# neutral_auds torch.Size([1, 512]), emotion_embed torch.Size([1, 16]) train
# neutral_auds torch.Size([171, 512]), emotion_embed torch.Size([1, 16]) test
 
        # emotion_embed = emotion_embed.repeat(neutral_auds.shape[0], 1) # only test
        # neutral_auds torch.Size([171, 512]), emotion_embed torch.Size([171, 16]) test

        emotion_embed = emotion_embed.unsqueeze(0) # neutral_auds torch.Size([1, 512]), emotion_embed torch.Size([1, 16])
  



        x = torch.cat([neutral_auds, emotion_embed], dim=1)  # [1,480]

        # print(f'x {x.shape}, emotion_embed {emotion_embed.shape}')
        # sys.exit()

        delta_auds = self.regressor(x)
        emotional_auds = neutral_auds + delta_scale * delta_auds

        # print(f'neutral_auds {neutral_auds.shape}, emotional_auds {emotional_auds.shape}')
        # sys.exit()


        return emotional_auds  #171,512
 