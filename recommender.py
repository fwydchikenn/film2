import torch
import numpy as np

def recommend_movies(
    model,
    user_sequence,
    item_embeddings,
    movies_df,
    k=10,
    max_seq_len=50
):
    model.eval()
    device = next(model.parameters()).device

    if len(user_sequence) > max_seq_len:
        user_sequence = user_sequence[-max_seq_len:]

    padded = [0] * (max_seq_len - len(user_sequence)) + user_sequence
    seq_tensor = torch.LongTensor([padded]).to(device)

    llm_features = []
    for item_id in padded:
        if item_id == 0:
            llm_features.append(np.zeros(768))
        else:
            llm_features.append(item_embeddings.get(item_id, np.zeros(768)))

    llm_tensor = torch.FloatTensor([llm_features]).to(device)

    with torch.no_grad():
        seq_out = model(seq_tensor, llm_tensor)
        logits = torch.matmul(
            seq_out[:, -1, :],
            model.item_embedding.weight[1:].T
        )

    scores, items = torch.topk(logits, k=k + len(user_sequence), dim=1)

    results = []
    for item_id, score in zip(items[0].cpu().numpy(), scores[0].cpu().numpy()):
        if item_id in user_sequence:
            continue

        movie = movies_df[movies_df["item_id"] == item_id]
        if not movie.empty:
            movie = movie.iloc[0]
            results.append({
                "title": movie["title"],
                "genres": movie["genres"],
                "score": float(score)
            })

        if len(results) == k:
            break

    return results
