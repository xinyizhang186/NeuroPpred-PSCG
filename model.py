import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# token_Embedding + position_Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding.from_pretrained(self.position_encoding(max_len, d_model), freeze=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p = dropout)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)  # , device=x.device # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        return self.norm(self.tok_embed(x) + self.pos_embed(pos))

    @staticmethod
    def position_encoding(max_len, d_model):
        """
        Position encoding feature introduced in "Attention is all you need",
        the b is changed to 1000 for the short length of sequence.
        """
        b = 1000
        pos_encoding = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (b ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (b ** (2 * i / d_model)))
        return torch.FloatTensor(pos_encoding)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: (B,L,D)
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class Model(nn.Module):
    def __init__(self, protT5_dim=1024, max_len=50):
        super().__init__()
        emb_dim = 320  # 320 256
        n_classes = 2
        dropout = 0.3

        # ProtT5 projection
        self.protT5_block = nn.Sequential(nn.Linear(protT5_dim, emb_dim), nn.LayerNorm(emb_dim), nn.ReLU())  #

        # SS embedding
        self.ss_embedding = Embedding(9, 64, max_len)
        self.ss_proj = nn.Sequential(nn.Linear(64, emb_dim), nn.LayerNorm(emb_dim), nn.ReLU())  #

        # Bidirectional Cross Attention
        self.cross_attn_p2s = nn.MultiheadAttention(emb_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.cross_attn_s2p = nn.MultiheadAttention(emb_dim, num_heads=8, dropout=dropout, batch_first=True)

        # Fusion gate
        self.fusion_gate = nn.Linear(emb_dim * 2, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

        # GatedCNN
        self.Conv1 = nn.Conv1d(emb_dim, emb_dim // 2, kernel_size=3, dilation=1, padding='same')  # , padding='same'
        self.Gate1 = nn.Conv1d(emb_dim, emb_dim // 2, kernel_size=3, dilation=1, padding='same')
        self.Conv2 = nn.Conv1d(emb_dim // 2, emb_dim // 4, kernel_size=5, dilation=2, padding='same')  # , dilation=2
        self.Gate2 = nn.Conv1d(emb_dim // 2, emb_dim // 4, kernel_size=5, dilation=2, padding='same')
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Attention Pool
        self.pool = AttentionPooling(emb_dim // 4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim // 4, emb_dim // 4), nn.BatchNorm1d(emb_dim // 4), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(emb_dim // 4, n_classes))

    def forward(self, protT5_feat, ss, mask=None):
        # mask = None

        # projection
        protT5_feat = F.dropout(protT5_feat, p=0.2, training=self.training)
        prot_feat = self.protT5_block(protT5_feat)
        ss_feat = self.ss_proj(self.ss_embedding(ss))

        # Cross Attention  # mask需要是bool 类型
        prot_attn, _ = self.cross_attn_p2s(query=prot_feat, key=ss_feat, value=ss_feat, key_padding_mask=mask)
        ss_attn, _ = self.cross_attn_s2p(query=ss_feat, key=prot_feat, value=prot_feat, key_padding_mask=mask)

        # Gated fusion
        fusion = torch.cat([prot_attn, ss_attn], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion))
        x = gate * prot_attn + (1 - gate) * ss_attn

        x = self.norm(x)

        # GatedCNN
        x = x.permute(0, 2, 1)

        x = self.relu(self.Conv1(x) * torch.sigmoid(self.Gate1(x)))
        x = self.dropout(x)
        x = self.relu(self.Conv2(x) * torch.sigmoid(self.Gate2(x)))
        x = self.dropout(x)

        # Pool
        x = x.permute(0, 2, 1)
        x = self.pool(x)

        return self.classifier(x)

