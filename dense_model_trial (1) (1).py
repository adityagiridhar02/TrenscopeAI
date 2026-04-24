import requests
import pandas as pd
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import cosine_similarity

def filter_by_keyword_relevance(df, keyword, embedder, threshold=0.35):
    """
    Keeps only posts semantically related to the keyword
    """
    keyword_emb = embedder.encode(keyword).reshape(1, -1)

    similarities = []
    for emb in df["embedding"]:
        sim = cosine_similarity(
            emb.reshape(1, -1),
            keyword_emb
        )[0][0]
        similarities.append(sim)

    df = df.copy()
    df["similarity"] = similarities

    filtered = df[df["similarity"] >= threshold]

    return filtered



def fetch_reddit_posts(keyword, pages=10):
    headers = {
        "User-Agent": "trendscopeAI/1.0 (Academic Research Project)"
    }

    all_posts = []
    after = None

    for i in range(pages):
        url = f"https://www.reddit.com/search.json?q={keyword}&sort=top&t=all"
        if after:
            url += f"&after={after}"

        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            break

        data = r.json()
        children = data["data"]["children"]

        if not children:
            break

        for post in children:
            p = post["data"]
            all_posts.append({
                "post_id": p.get("id"),
                "text": p.get("title", ""),
                "upvotes": p.get("score", 0),
                "comments": p.get("num_comments", 0),
                "created_utc": p.get("created_utc"),
                "platform": "Reddit",
                "keyword": keyword
            })

        after = data["data"].get("after")
        if not after:
            break

        time.sleep(1)

    return pd.DataFrame(all_posts)


def aggregate_daily(df, embedder):
    if df.empty:
        return None, None

    df["date"] = df["created_utc"].apply(
        lambda x: datetime.datetime.fromtimestamp(x).date()
    )

    df = df[df["text"].str.len() > 10]

    df = df.copy()
    df.loc[:, "embedding"] = df["text"].apply(lambda x: embedder.encode(x))


    daily_vectors = []
    daily_meta = []

    for date, g in df.groupby("date"):
        text_emb = np.mean(np.vstack(g["embedding"]), axis=0)

        engagement = np.array([
            g["upvotes"].mean(),
            g["comments"].mean()
        ])

        vector = np.concatenate([text_emb, engagement])

        daily_vectors.append(vector)
        daily_meta.append({
            "date": date,
            "posts": len(g)
        })

    X = np.vstack(daily_vectors)
    return X, daily_meta

 
 
class TemporalSSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out)

 
 
class LifespanPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

 
 
def summarize_trend(df):
    top = df.sort_values("upvotes", ascending=False).head(5)
    bullets = []
    for t in top["text"]:
        bullets.append("- " + t[:120] + ("..." if len(t) > 120 else ""))
    return "\n".join(bullets)

 
 
def run_trend_lifespan_demo(keyword):
    print(f"\n  Running Trend Analysis for: {keyword}\n")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    df = fetch_reddit_posts(keyword, pages=10)

    if df.empty or len(df) < 10:
        print("  Not enough Reddit data to analyze this trend.")
        return
    X, meta = aggregate_daily(df, embedder)
    df = df.copy()
    df.loc[:, "embedding"] = df["text"].apply(lambda x: embedder.encode(x))

    df = filter_by_keyword_relevance(df, keyword, embedder, threshold=0.35)

    if df.empty or len(df) < 10:
        print("  Not enough semantically relevant posts.")
        return

    X, meta = aggregate_daily(df, embedder)

    if X is None or X.shape[0] < 3:
        print("  Not enough daily data points.")
        return

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    x_seq = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    encoder = TemporalSSM(x_seq.shape[-1])
    head = LifespanPredictor(x_seq.shape[-1])

    with torch.no_grad():
        h = encoder(x_seq)
        final_state = h[:, -1, :]
    print("Trend Summary:")
    print(summarize_trend(df))

    print(f"\nObserved Active Days: {len(meta)}")

 
if __name__ == "__main__":
    keyword = input("Enter trend keyword: ")
    run_trend_lifespan_demo(keyword)
