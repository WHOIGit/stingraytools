import argparse
import hashlib
import logging
import os
import shutil
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from stingray.logging.setup import log_command_options, setup_logging

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

log = logging.getLogger(__name__)
SLURM_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
DEFAULT_NUM_WORKERS = max(1, SLURM_CPUS - 1 if SLURM_CPUS > 1 else 1)


def warn(message):
    log.warning(message)

# ===============================
# Parse arguments
# ===============================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="stingray images generate-training",
        description="Prepare YOLO training data from frame_states.csv, rois.csv and class_dict.csv annotation exports.",
        epilog=(
            "Template image dimensions: --img-dim 2330 1750. "
            "For Slurm, use --no-confirm --no-date-suffix and pass an exact --out-dir."
        ),
    )
    p.add_argument("--in-dir", dest="in_dir", type=str, nargs="+", required=True, help="Multiple input directories")
    p.add_argument("--img-dim", dest="img_dim", type=int, nargs=2, required=True, help="Image width height")
    p.add_argument("--out-dir", dest="out_dir", type=str, required=True)
    p.add_argument(
        "--no-date-suffix",
        action="store_true",
        help="Use --out-dir exactly instead of appending today's date.",
    )
    p.add_argument("--random-state", dest="random_state", type=int, required=True)
    p.add_argument("--val-ratio", dest="val_ratio", type=float, required=True)
    p.add_argument("--min-val-per-class", dest="min_val_per_class", type=int, required=True)
    p.add_argument("--area", type=float, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--extensions", type=str, nargs="+", default=[".png"])
    p.add_argument("--num-workers", dest="num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-yaml", dest="save_yaml", action="store_true")
    p.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the interactive confirmation prompt for batch or Slurm runs.",
    )
    p.add_argument(
        "--work-dir",
        default=".",
        help="Workspace whose logs directory receives command logs.",
    )
    p.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable Stingray log files and write logs only to the console.",
    )
    return p.parse_args(argv)

# ===============================
# File load + merge (keep latest)
# ===============================
def load_and_merge(csv_name, in_dirs, header="infer"):
    dfs = []
    for d in in_dirs:
        path = os.path.join(d, csv_name)
        if os.path.exists(path):
            df = pd.read_csv(path, header=header)
            df["__source__"] = d
            dfs.append(df)
        else:
            warn(f"{csv_name} missing in {d}")
    if not dfs:
        raise FileNotFoundError(f"Required CSV {csv_name} missing in all input folders")

    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(keep="last").reset_index(drop=True)
    log.info(f"{csv_name}: merged {before} -> {len(merged)} rows")
    return merged

# ===============================
# MD5 checksum
# ===============================
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ===============================
# Copy images
# ===============================
def find_image(filename, in_dirs):
    for d in in_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

def copy_image(row, split, in_dirs, out_dir, exts, overwrite):
    out_img_dir = os.path.join(out_dir, split, "images")
    for ext in exts:
        fname = f"{row.media_id}_{row.frame}{ext}"
        src = find_image(fname, in_dirs)
        if src:
            dst = os.path.join(out_img_dir, f"{row.image}{ext}")
            if os.path.exists(dst) and not overwrite:
                # compare checksum
                if md5(src) == md5(dst):
                    return
            shutil.copyfile(src, dst)
            return
    warn(f"Image not found: {row.media_id}_{row.frame}")

def copy_images(df, split, in_dirs, out_dir, num_workers, exts, overwrite, verbose):
    rows = list(df.drop_duplicates("image").itertuples())
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(copy_image, r, split, in_dirs, out_dir, exts, overwrite) for r in rows]
        if tqdm and verbose:
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Copy [{split}]"):
                _.result()
        else:
            for f in as_completed(futures): f.result()

# ===============================
# Write labels
# ===============================
def write_label(image, group, split, out_dir, overwrite):
    path = os.path.join(out_dir, split, "labels", f"{image}.txt")
    if os.path.exists(path) and not overwrite:
        return
    with open(path, "w") as f:
        f.write("\n".join(group["yolo"].dropna()) + "\n")

def write_labels(df, split, out_dir, num_workers, overwrite, verbose):
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(write_label, img, grp, split, out_dir, overwrite)
                   for img, grp in df.groupby("image")]
        if tqdm and verbose:
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Labels [{split}]"):
                _.result()
        else:
            for f in as_completed(futures): f.result()

# ===============================
# Split train/val
# ===============================
def split_data(df, ratio, min_per_class, seed):
    rng = np.random.default_rng(seed)
    imgs = sorted(df["image"].unique())
    groups = {img: df[df.image == img] for img in imgs}
    cls_to_imgs = defaultdict(list)
    for img in imgs:
        for c in groups[img]["class_train"].dropna().unique():
            cls_to_imgs[c].append(img)

    val = set()
    total_roi = len(df)
    target_roi = int(total_roi * ratio)

    # ensure minimum per class
    for cls in sorted(cls_to_imgs):
        cand = cls_to_imgs[cls]
        rng.shuffle(cand)
        count = 0
        for img in cand:
            if img in val: continue
            val.add(img)
            count += (groups[img]["class_train"] == cls).sum()
            if count >= min_per_class:
                break

    # fill remainder
    need = target_roi - sum(len(groups[i]) for i in val)
    if need > 0:
        rem = sorted(set(imgs) - val)
        rng.shuffle(rem)
        acc = 0
        for img in rem:
            val.add(img)
            acc += len(groups[img])
            if acc >= need: break

    df2 = df.copy()
    df2["split"] = df2["image"].apply(lambda x: "val" if x in val else "train")
    return df2[df2.split=="train"].reset_index(drop=True), df2[df2.split=="val"].reset_index(drop=True)

# ===============================
# YAML
# ===============================
def write_yaml(classes, out_dir):
    p = os.path.join(out_dir, "data.yaml")
    with open(p,"w") as f:
        f.write("train:\n")
        f.write(f" - {os.path.join(out_dir, 'train/images')}\n")
        f.write("val:\n")
        f.write(f" - {os.path.join(out_dir, 'val/images')}\n")
        f.write("nc: " + str(len(classes)) + "\nnames:\n")
        for i,c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

# ===============================
# MAIN
# ===============================
def main(argv=None):
    args = parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    setup_logging(
        log_dir=work_dir / "logs",
        name="stingray_images_generate_yolo_training",
        file=not args.no_file_log,
    )
    log_command_options(log, args)

    # Keep dated output directories for manual runs unless batch mode asks for an exact path.
    if args.no_date_suffix:
        final_out = args.out_dir.rstrip("/")
    else:
        today = datetime.now().strftime("%Y%m%d")
        outdir = args.out_dir.rstrip("/") + "_" + today
        i = 1
        final_out = outdir
        while os.path.exists(final_out):
            final_out = f"{outdir}_{i}"
            i += 1
    args.out_dir = final_out
    log.info(f"Output directory: {args.out_dir}")

    # load CSVs
    framestate = load_and_merge("frame_states.csv", args.in_dir)
    rois       = load_and_merge("rois.csv", args.in_dir)
    classdict  = load_and_merge("class_dict.csv", args.in_dir, header=None)

    # prep ROI
    if "Verified" not in rois.columns:
        warn("'Verified' missing in rois → set True")
        rois["Verified"] = True
    rois = rois[rois.Verified==True].reset_index(drop=True)
    rois["class_corr"] = [
        c if not pd.isna(c) else o
        for c,o in zip(rois.get("ClassStrCorrection"), rois.get("Class"))
    ]

    # framestate
    for col in ["Verified","training","holdout"]:
        if col not in framestate.columns:
            framestate[col] = (col!="holdout")
    framestate = framestate[(framestate.Verified)&(framestate.training)&(~framestate.holdout)]

    # merge
    df = pd.merge(framestate, rois, on=["media_id","frame"], how="left")
    df["image"] = df.apply(lambda r: f"{os.path.splitext(str(r.media_id))[0]}_{r.frame}", axis=1)

    # class dict
    taxa = dict(zip(classdict[0],classdict[1]))
    df["class_train"] = df["class_corr"].map(taxa)
    df = df[(df.class_train!="remove_label") & (df.class_train!="remove_frame")]
    df["class_train"] = df["class_train"].fillna("_background_")

    # classes
    classes = sorted(x for x in df.class_train.unique() if x!="_background_")
    class_id = {c:i for i,c in enumerate(classes)}
    df["class_id"] = df["class_train"].map(class_id)

    # YOLO
    W,H = args.img_dim
    df["x_center"] = df.x + df.width/2
    df["y_center"] = df.y + df.height/2
    df["area"] = df.width*W * df.height*H
    # REMOVE background labels so no NaN labels are written
    df = df[df["class_train"] != "_background_"].copy()
    df["yolo"] = df.apply(lambda r: f"{r.class_id} {r.x_center:.6f} {r.y_center:.6f} {r.width:.6f} {r.height:.6f}", axis=1)

    # filter area
    df = df[(df.area>=args.area) | df.area.isna()]

    # split
    train, val = split_data(df, args.val_ratio, args.min_val_per_class, args.random_state)

    # --- Summarize split ---
    total_images = df['image'].nunique()
    train_images = train['image'].nunique()
    val_images = val['image'].nunique()
    train_counts = train['class_train'].value_counts()
    val_counts = val['class_train'].value_counts()
    total_counts = df['class_train'].value_counts()

    summary_df = pd.DataFrame({
        'Total count': total_counts,
        'Train': train_counts,
        'Train %': train_counts / total_counts,
        'Validation': val_counts,
        'Validation %': val_counts / total_counts,
    }).fillna(0)
    summary_df.insert(0, 'ID', summary_df.index.map(class_id).astype('Int64'))

    log.info(f"\nTotal images: {total_images}")
    log.info(f"Train images: {train_images} ({train_images / total_images:.2%})")
    log.info(f"Val images:   {val_images} ({val_images / total_images:.2%})")
    log.info("\nClass Distribution Summary:")
    log.info(summary_df.to_string(formatters={
        'Train %': '{:.2%}'.format,
        'Validation %': '{:.2%}'.format
    }))

    # Require manual confirmation unless the caller explicitly opts into batch mode.
    if not args.no_confirm:
        ans = input("Continue? (y/n): ").lower().strip()
        if ans!="y":
            log.info("Cancelled")
            sys.exit(0)

    # create dirs
    for s in ["train","val"]:
        os.makedirs(os.path.join(args.out_dir,s,"images"),exist_ok=True)
        os.makedirs(os.path.join(args.out_dir,s,"labels"),exist_ok=True)

    # write labels
    write_labels(train,"train",args.out_dir,args.num_workers,args.overwrite,args.verbose)
    write_labels(val,"val",args.out_dir,args.num_workers,args.overwrite,args.verbose)

    # copy images
    copy_images(train,"train",args.in_dir,args.out_dir,args.num_workers,args.extensions,args.overwrite,args.verbose)
    copy_images(val,"val",args.in_dir,args.out_dir,args.num_workers,args.extensions,args.overwrite,args.verbose)

    # save csv
    train.to_csv(os.path.join(args.out_dir,"train_labels.csv"),index=False)
    val.to_csv(os.path.join(args.out_dir,"val_labels.csv"),index=False)

    # yaml
    if args.save_yaml:
        write_yaml(classes,args.out_dir)

    log.info("DONE.")

if __name__ == "__main__":
    main()
