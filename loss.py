'''
Author: Samrito
Date: 2022-11-10 20:34:22
LastEditors: Samrito
LastEditTime: 2022-12-05 16:20:11
'''
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers.SentenceTransformer import SentenceTransformer


class CoSENTLoss(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 cos_score_transformation=nn.Identity()):
        super(CoSENTLoss, self).__init__()
        self.model = model
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]],
                labels: Tensor):
        embeddings = [
            self.model(sentence_feature)['sentence_embedding']
            for sentence_feature in sentence_features
        ]
        output = self.cos_score_transformation(
            torch.cosine_similarity(embeddings[0], embeddings[1]))
        output = 20 * output  # 20 is hyper-parameter
        output = output[:, None] - output[None, :]  # (i-cos) - (j-cos)
        labels = labels[:, None] < labels[None, :]  # i, j, i neg, j pos , 1
        labels = labels.float()
        output = output - (1 - labels) * 1e12
        # output = output - (1 - labels) * 1e20
        output = torch.cat((torch.zeros(1).to(output.device), output.view(-1)),
                           dim=0)
        loss = torch.logsumexp(output, dim=0)

        return loss
