import torch
import torch.nn as nn

class SASRecWithLLMAndIPS(nn.Module):
    def __init__(
        self,
        n_items,
        embedding_dim=128,
        n_heads=2,
        n_layers=2,
        max_seq_len=50,
        dropout=0.2,
        llm_dim=768
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.llm_projection = nn.Linear(llm_dim, embedding_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embedding_dim, n_heads, dropout=dropout, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 4, embedding_dim),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])

        self.norm1 = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(n_layers)
        ])
        self.norm2 = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq, llm_features):
        batch_size, seq_len = item_seq.shape

        seq_emb = self.item_embedding(item_seq)
        llm_emb = self.llm_projection(llm_features)

        gate = self.fusion_gate(torch.cat([seq_emb, llm_emb], dim=-1))
        seq_emb = gate * seq_emb + (1 - gate) * llm_emb

        pos = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        seq_emb = seq_emb + self.pos_embedding(pos)
        seq_emb = self.dropout(seq_emb)

        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device) * float('-inf'),
            diagonal=1
        )

        for i in range(len(self.attention_layers)):
            residual = seq_emb
            seq_emb = self.norm1[i](seq_emb)
            seq_emb, _ = self.attention_layers[i](
                seq_emb, seq_emb, seq_emb, attn_mask=attn_mask
            )
            seq_emb = residual + seq_emb

            residual = seq_emb
            seq_emb = self.norm2[i](seq_emb)
            seq_emb = self.ffn_layers[i](seq_emb)
            seq_emb = residual + seq_emb

        return seq_emb
