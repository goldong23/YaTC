import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split a recursive image dataset into train/test directories for YaTC.'
    )
    parser.add_argument('--input-root', required=True, help='Source dataset root, e.g. /path/to/My_MFR_Dataset')
    parser.add_argument('--output-root', required=True, help='Output dataset root containing train/ and test/')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Fraction of groups assigned to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splitting')
    parser.add_argument(
        '--label-level',
        type=int,
        default=1,
        help='1-based directory depth under input-root used as the class label',
    )
    parser.add_argument(
        '--group-level',
        type=int,
        default=2,
        help='1-based directory depth under input-root used as the atomic split unit',
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying them',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned split without creating files',
    )
    return parser.parse_args()


def iter_image_files(root: Path):
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def collect_groups(input_root: Path, label_level: int, group_level: int):
    if label_level < 1:
        raise ValueError(f'label_level must be >= 1, got {label_level}')
    if group_level < label_level:
        raise ValueError(
            f'group_level must be >= label_level, got label_level={label_level}, group_level={group_level}'
        )

    grouped = defaultdict(list)
    for path in iter_image_files(input_root):
        rel_parts = path.relative_to(input_root).parts
        parent_parts = rel_parts[:-1]
        if len(parent_parts) < group_level:
            raise RuntimeError(
                f'Found file shallower than group_level={group_level}: {path}'
            )
        label = parent_parts[label_level - 1]
        group_key = parent_parts[:group_level]
        grouped[(label, group_key)].append(path)

    if not grouped:
        raise RuntimeError(f'No image files found under {input_root}')

    return grouped


def choose_test_groups(grouped, test_ratio: float, seed: int):
    rng = random.Random(seed)
    groups_by_label = defaultdict(list)
    for label, group_key in grouped:
        groups_by_label[label].append(group_key)

    split_map = {}
    for label, group_keys in groups_by_label.items():
        unique_keys = sorted(set(group_keys))
        rng.shuffle(unique_keys)
        n_test = int(round(len(unique_keys) * test_ratio))
        if len(unique_keys) > 1:
            n_test = max(1, min(len(unique_keys) - 1, n_test))
        else:
            n_test = 0

        test_keys = set(unique_keys[:n_test])
        for key in unique_keys:
            split_map[(label, key)] = 'test' if key in test_keys else 'train'

    return split_map


def copy_or_move_files(grouped, split_map, input_root: Path, output_root: Path, move: bool):
    action = shutil.move if move else shutil.copy2
    counts = defaultdict(int)

    for grouped_key, files in grouped.items():
        split = split_map[grouped_key]
        for src in files:
            rel = src.relative_to(input_root)
            dst = output_root / split / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            action(src, dst)
            counts[split] += 1

    return counts


def print_summary(grouped, split_map):
    counts = defaultdict(lambda: defaultdict(int))
    group_counts = defaultdict(lambda: defaultdict(int))

    for grouped_key, files in grouped.items():
        label, _ = grouped_key
        split = split_map[grouped_key]
        counts[split][label] += len(files)
        group_counts[split][label] += 1

    for split in ['train', 'test']:
        total_groups = sum(group_counts[split].values())
        total_files = sum(counts[split].values())
        print(f'[{split}] groups={total_groups}, files={total_files}')
        for label in sorted(counts[split]):
            print(
                f'  - {label}: groups={group_counts[split][label]}, files={counts[split][label]}'
            )


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if input_root == output_root:
        raise ValueError('input-root and output-root must be different')
    if not input_root.is_dir():
        raise FileNotFoundError(f'Input dataset directory does not exist: {input_root}')
    if output_root.exists() and any(output_root.iterdir()) and not args.dry_run:
        raise RuntimeError(
            f'Output directory already exists and is not empty: {output_root}'
        )

    grouped = collect_groups(input_root, args.label_level, args.group_level)
    split_map = choose_test_groups(grouped, args.test_ratio, args.seed)
    print_summary(grouped, split_map)

    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    counts = copy_or_move_files(grouped, split_map, input_root, output_root, args.move)
    print(f"Copied files: train={counts['train']}, test={counts['test']}")


if __name__ == '__main__':
    main()
