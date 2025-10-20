import argparse, json, os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from seed_utils import set_all_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to CSV with columns: id,label")
    parser.add_argument("--preds", required=True, help="Path to CSV with columns: id,pred")
    args = parser.parse_args()

    set_all_seeds(42)

    # Load files
    ytrue = pd.read_csv(args.labels)
    ypred = pd.read_csv(args.preds)

    # Align by ID so both files match
    merged = ytrue.merge(ypred, on="id", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping ids between labels and preds.")

    # Compute metrics
    acc = accuracy_score(merged["label"], merged["pred"])
    f1  = f1_score(merged["label"], merged["pred"], average="macro")

    # Save results
    os.makedirs("results", exist_ok=True)
    out = {"accuracy": round(acc, 6), "f1_macro": round(f1, 6), "n": int(len(merged))}
    with open("results/results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Accuracy:", out["accuracy"])
    print("F1_macro:", out["f1_macro"])
    print("N Samples:", out["n"])
    print("Saved to results/results.json")

if __name__ == "__main__":
    main()
