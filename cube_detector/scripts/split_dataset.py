"""
split_dataset.py
Splits positives_raw (with labels) and negatives_raw (no labels)
into dataset/images/{train,val} and dataset/labels/{train,val}.
Negatives get empty .txt label files automatically.

Usage:
    python split_dataset.py --val_ratio 0.15
"""
import os, shutil, random, argparse

def collect_pairs(raw_dir, is_negative=False):
    pairs = []
    for f in os.listdir(raw_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(f)[0]
            img_path = os.path.join(raw_dir, f)
            label_path = os.path.join(raw_dir, name + ".txt")
            if is_negative:
                pairs.append((img_path, None, f, name + ".txt"))
            else:
                if not os.path.exists(label_path):
                    print(f"WARNING: no label for {f}, skipping (label it first)")
                    continue
                pairs.append((img_path, label_path, f, name + ".txt"))
    return pairs

def write_split(pairs, img_dir, label_dir):
    for img_path, label_path, img_name, label_name in pairs:
        shutil.copy(img_path, os.path.join(img_dir, img_name))
        if label_path is None:
            open(os.path.join(label_dir, label_name), "w").close()  # empty file
        else:
            shutil.copy(label_path, os.path.join(label_dir, label_name))

def main(val_ratio):
    pos = collect_pairs("dataset/positives_raw", is_negative=False)
    neg = collect_pairs("dataset/negatives_raw", is_negative=True)
    all_pairs = pos + neg
    random.shuffle(all_pairs)

    n_val = int(len(all_pairs) * val_ratio)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    write_split(train_pairs, "dataset/images/train", "dataset/labels/train")
    write_split(val_pairs, "dataset/images/val", "dataset/labels/val")

    print(f"Positives: {len(pos)}, Negatives: {len(neg)}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_ratio", type=float, default=0.15)
    args = p.parse_args()
    main(args.val_ratio)