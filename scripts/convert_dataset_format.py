import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
import yaml


def filter_and_sample_coco_images(
    json_path: Path,
    max_images: int = 2500,
    seed: int = 42,
) -> tuple[dict, dict, dict, dict]:
    """Filter annotated images from COCO JSON and sample by split ratio."""
    random.seed(seed)

    if not json_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {json_path}")

    print(f"Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Map categories to contiguous class indices
    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    cat_id_to_cls = {cat["id"]: i for i, cat in enumerate(categories)}
    class_names = {i: cat["name"] for i, cat in enumerate(categories)}

    # Collect valid piece annotations grouped by image_id
    anns_data = coco.get("annotations", {})
    pieces_list = (
        anns_data.get("pieces", [])
        if isinstance(anns_data, dict)
        else (anns_data if isinstance(anns_data, list) else [])
    )

    img_to_anns = defaultdict(list)
    for ann in pieces_list:
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if cat_id in cat_id_to_cls and bbox and len(bbox) == 4:
            img_to_anns[ann["image_id"]].append(ann)

    # Filter images with at least 1 annotation
    all_images = coco.get("images", [])
    annotated_img_ids = set(img_to_anns.keys())
    valid_images = [img for img in all_images if img["id"] in annotated_img_ids]

    # Map images to dataset splits
    splits_info = coco.get("splits", {})
    split_name_map = {"train": "train", "val": "valid", "test": "test"}

    split_to_imgs = defaultdict(list)
    for img in valid_images:
        img_id = img["id"]
        found_split = None
        for s_key, s_val in splits_info.items():
            if img_id in s_val.get("image_ids", []):
                found_split = split_name_map.get(s_key, s_key)
                break
        if not found_split:
            found_split = "train"
        split_to_imgs[found_split].append(img)

    total_valid = sum(len(imgs) for imgs in split_to_imgs.values())
    print(f"Total images with annotations: {total_valid}")

    # Sample images proportional to original split sizes
    if total_valid > max_images:
        sampled_to_imgs = {}
        splits_order = ["train", "valid", "test"]
        active_splits = [s for s in splits_order if s in split_to_imgs]

        counts = {}
        running_total = 0
        for idx, s in enumerate(active_splits):
            if idx == len(active_splits) - 1:
                counts[s] = max_images - running_total
            else:
                ratio = len(split_to_imgs[s]) / total_valid
                count = int(round(max_images * ratio))
                counts[s] = count
                running_total += count

        for s in active_splits:
            sampled_to_imgs[s] = random.sample(split_to_imgs[s], counts[s])
    else:
        sampled_to_imgs = split_to_imgs

    for s, imgs in sampled_to_imgs.items():
        print(f"Split '{s}': {len(imgs)} images selected")

    return sampled_to_imgs, img_to_anns, cat_id_to_cls, class_names


def export_to_yolo_format(
    sampled_to_imgs: dict,
    img_to_anns: dict,
    cat_id_to_cls: dict,
    class_names: dict,
    images_dir: Path,
    output_dir: Path,
) -> None:
    """Export filtered image annotations to YOLO format directory structure."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for split, imgs in sampled_to_imgs.items():
        split_img_dir = output_dir / split / "images"
        split_lbl_dir = output_dir / split / "labels"

        for img_info in imgs:
            img_id = img_info["id"]
            h, w = img_info["height"], img_info["width"]
            rel_path = img_info.get("path", img_info.get("file_name", ""))

            # Resolve image file location
            src_img_file = images_dir.parent / rel_path
            if not src_img_file.exists():
                src_img_file = images_dir / rel_path
            if not src_img_file.exists() and rel_path.startswith("images/"):
                src_img_file = images_dir / rel_path[len("images/"):]
            if not src_img_file.exists():
                src_img_file = images_dir / Path(rel_path).name

            if not src_img_file.exists():
                print(f"Warning: Image file not found: {rel_path}, skipping.")
                continue

            file_name = Path(rel_path).name
            dst_img_file = split_img_dir / file_name
            shutil.copy2(src_img_file, dst_img_file)

            # Generate YOLO label file (.txt)
            txt_file = split_lbl_dir / f"{dst_img_file.stem}.txt"
            label_lines = []

            for ann in img_to_anns.get(img_id, []):
                cls_idx = cat_id_to_cls[ann["category_id"]]
                x_top, y_top, bw, bh = ann["bbox"]

                # Convert bbox to normalized center coordinates
                xc = (x_top + bw / 2.0) / w
                yc = (y_top + bh / 2.0) / h
                nw = bw / w
                nh = bh / h

                label_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            with open(txt_file, "w") as lf:
                lf.write("\n".join(label_lines) + ("\n" if label_lines else ""))

            copied_count += 1

    # Write data.yaml file
    data_yaml = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(output_dir / "data.yaml", "w") as yf:
        yaml.dump(data_yaml, yf, default_flow_style=False, sort_keys=False)

    print(f"\nDataset conversion complete! Output saved to: {output_dir}")
    print(f"Total converted images: {copied_count}")


def extract_unannotated_images(
    json_path: Path,
    images_dir: Path,
    no_annotation_dir: Path,
) -> None:
    """Extract images without valid annotations to a separate directory."""
    if not json_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {json_path}")

    print(f"Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
    cat_ids = set(cat["id"] for cat in categories)

    anns_data = coco.get("annotations", {})
    pieces_list = (
        anns_data.get("pieces", [])
        if isinstance(anns_data, dict)
        else (anns_data if isinstance(anns_data, list) else [])
    )

    annotated_img_ids = set()
    for ann in pieces_list:
        cat_id = ann.get("category_id")
        bbox = ann.get("bbox")
        if cat_id in cat_ids and bbox and len(bbox) == 4:
            annotated_img_ids.add(ann["image_id"])

    all_images = coco.get("images", [])
    unannotated_imgs = [img for img in all_images if img["id"] not in annotated_img_ids]
    print(f"Total unannotated images found: {len(unannotated_imgs)}")

    no_annotation_dir = no_annotation_dir.resolve()
    no_annotation_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for img_info in unannotated_imgs:
        rel_path = img_info.get("path", img_info.get("file_name", ""))

        src_img_file = images_dir.parent / rel_path
        if not src_img_file.exists():
            src_img_file = images_dir / rel_path
        if not src_img_file.exists() and rel_path.startswith("images/"):
            src_img_file = images_dir / rel_path[len("images/"):]
        if not src_img_file.exists():
            src_img_file = images_dir / Path(rel_path).name

        if not src_img_file.exists():
            print(f"Warning: Image file not found: {rel_path}, skipping.")
            continue

        file_name = Path(rel_path).name
        dst_img_file = no_annotation_dir / file_name
        shutil.copy2(src_img_file, dst_img_file)
        copied_count += 1

    print(f"Extraction complete! Saved {copied_count} unannotated images to: {no_annotation_dir}")


def convert_chessred_dataset(
    json_path: Path,
    images_dir: Path,
    output_dir: Path,
    max_images: int = 2500,
    seed: int = 42,
) -> None:
    """Main workflow to filter COCO annotations and export to YOLO dataset format."""
    sampled_to_imgs, img_to_anns, cat_id_to_cls, class_names = filter_and_sample_coco_images(
        json_path=json_path,
        max_images=max_images,
        seed=seed,
    )

    export_to_yolo_format(
        sampled_to_imgs=sampled_to_imgs,
        img_to_anns=img_to_anns,
        cat_id_to_cls=cat_id_to_cls,
        class_names=class_names,
        images_dir=images_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset format or filter unannotated images.")
    parser.add_argument(
        "--no-annotation-only",
        action="store_true",
        help="Extract images without annotations into a separate directory and exit.",
    )
    args = parser.parse_args()

    dataset_dir = Path("/home/duykhongcay/lerobot_ws/chess_pieces_detection/datasets")
    json_path = dataset_dir / "annotations.json"
    images_dir = dataset_dir / "images"
    output_dir = dataset_dir / "chessred"
    no_annotation_dir = dataset_dir / "no_annotation"

    if args.no_annotation_only:
        extract_unannotated_images(
            json_path=json_path,
            images_dir=images_dir,
            no_annotation_dir=no_annotation_dir,
        )
    else:
        convert_chessred_dataset(
            json_path=json_path,
            images_dir=images_dir,
            output_dir=output_dir,
            max_images=2500,
        )

