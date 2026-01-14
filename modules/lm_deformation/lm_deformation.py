import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class LandmarkDeformationModel(nn.Module):
    def __init__(self, landmark_dim=204, num_emotions=8, emotion_embedding_dim=16, hidden_dim=256):
        super().__init__()

        # Emotion embedding
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)

        # Input dimension: neutral landmarks + emotion embedding
        input_dim = landmark_dim + emotion_embedding_dim

        # Regression network
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, landmark_dim)
        )
    

    def forward(self, neutral_landmarks, emotion_labels, delta=1):
        """
        neutral_landmarks: Tensor of shape [B, landmark_dim]
        emotion_labels: Tensor of shape [B] (LongTensor)
        """
        # print(f'neutral_landmarks {neutral_landmarks.shape}, emotion_labels {emotion_labels}')

        emotion_embed = self.emotion_embedding(emotion_labels)  
        if emotion_embed.dim() == 1:
            emotion_embed = emotion_embed.unsqueeze(0)# [1,16]
        if emotion_embed.dim() == 3:
            emotion_embed = emotion_embed.squeeze(1) # [1,16]

        if neutral_landmarks.shape[0] != emotion_embed.shape[0]:
            emotion_embed = emotion_embed.repeat(neutral_landmarks.shape[0], 1)

# neutral_landmarks torch.Size([240, 204]), emotion_labels torch.Size([1, 1]) test
# neutral_landmarks torch.Size([1, 204]), emotion_labels [] train
        # emotion_embed = self.emotion_embedding(emotion_labels) 
# neutral_landmarks torch.Size([240, 204]), emotion_embed torch.Size([1, 1, 16]) test
# neutral_landmarks torch.Size([1, 204]) , emotion_embed torch.Size([16]) train
        # emotion_embed = emotion_embed.squeeze(1) # unsqueeze(0) in train, squeeze(1) in test
        # emotion_embed = emotion_embed.repeat(neutral_landmarks.shape[0], 1)
# neutral_landmarks torch.Size([240, 204]), emotion_embed torch.Size([240, 16]) test
# neutral_landmarks torch.Size([1, 204]) , emotion_embed torch.Size([1, 16]) train



        x = torch.cat([neutral_landmarks, emotion_embed], dim=1)  # [B, landmark_dim + emotion_embedding_dim]

        # Predict landmark deformation
        delta_landmarks = self.regressor(x)  # [B, landmark_dim]

        # Compute emotional landmarks
        emotional_landmarks = neutral_landmarks + delta_landmarks * delta  

        return emotional_landmarks
