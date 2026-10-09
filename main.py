"""Tiny embedding similarity search utility."""
import sys, math, argparse


def parse_embeddings(path):
    """Read embeddings file: each line 'label v1 v2 ...'."""
    emb = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            label = parts[0]
            vec = list(map(float, parts[1:]))
            emb[label] = vec
    return emb


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def main():
    parser = argparse.ArgumentParser(description="Find nearest embeddings")
    parser.add_argument("file", help="Embeddings file")
    parser.add_argument("query", nargs="+", help="Query vector components")
    parser.add_argument("--top", "-t", type=int, default=5, help="Number of top results")
    args = parser.parse_args()

    emb = parse_embeddings(args.file)
    query = list(map(float, args.query))
    sims = [(label, cosine_similarity(vec, query)) for label, vec in emb.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    for label, sim in sims[:args.top]:
        print(f"{label}\t{sim:.4f}")


if __name__ == "__main__":
    main()