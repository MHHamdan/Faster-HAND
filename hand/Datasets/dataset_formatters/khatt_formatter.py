#  KHATT Arabic Handwriting Database Formatter for HAND framework
#  Formats KHATT dataset for OCR/HTR training at line or paragraph level

from hand.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import os
import numpy as np
from PIL import Image
from collections import defaultdict


class KHATTDatasetFormatter(OCRDatasetFormatter):
    """
    Formatter for the KHATT (KFUPM Handwritten Arabic TexT) Database.

    KHATT is an Arabic handwriting dataset with:
    - 1000 writers from different countries
    - Line-level images with manually verified transcriptions
    - ~6673 line samples (4672 train, 963 valid, 1038 test)
    - ~2693 unique paragraphs (lines can be grouped into paragraphs)

    Dataset structure:
    KHATT/
    ├── images/           # Line images (*.jpg)
    ├── labels/           # Transcriptions (*.txt, cp1256 encoded)
    ├── train_ids.txt     # Training sample IDs
    ├── val_ids.txt       # Validation sample IDs
    ├── test_ids.txt      # Test sample IDs
    └── syms.txt          # Character set

    Levels:
    - line: Individual text lines (for CTC training)
    - paragraph: Lines grouped by paragraph ID (for HAND seq2seq training)
    """

    def __init__(self, level="line", set_names=["train", "valid", "test"], dpi=150,
                 source_dpi=300, raw_data_path=None):
        """
        Args:
            level: 'line' or 'paragraph'
            set_names: dataset splits to create
            dpi: target DPI for output images
            source_dpi: source DPI of KHATT images (assumed 300 dpi)
            raw_data_path: path to raw KHATT data
        """
        super(KHATTDatasetFormatter, self).__init__("KHATT", level, "", set_names)

        self.dpi = dpi
        self.source_dpi = source_dpi

        if raw_data_path:
            self.source_fold_path = raw_data_path

        # Label encoding for KHATT (Windows Arabic)
        self.label_encoding = 'cp1256'

        # Split files mapping
        self.split_files = {
            "train": "train_ids.txt",
            "valid": "val_ids.txt",
            "test": "test_ids.txt"
        }

        self.map_datasets_files.update({
            "KHATT": {
                "line": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_khatt_line,
                },
                "paragraph": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_khatt_paragraph,
                },
            }
        })

    def init_format(self):
        """Initialize formatting - create output directories"""
        os.makedirs(self.target_fold_path, exist_ok=True)
        for set_name in self.set_names:
            os.makedirs(os.path.join(self.target_fold_path, set_name), exist_ok=True)

    def end_format(self):
        """Save labels.pkl file"""
        import pickle
        with open(os.path.join(self.target_fold_path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": self.gt,
                "charset": sorted(list(self.charset)),
            }, f)
        print(f"Saved labels.pkl with {len(self.charset)} characters")
        for set_name in self.set_names:
            print(f"  {set_name}: {len(self.gt[set_name])} samples")

    def load_split_ids(self, split_name):
        """Load sample IDs for a split from the split file"""
        split_file = self.split_files.get(split_name)
        if not split_file:
            return []

        split_path = os.path.join(self.source_fold_path, split_file)
        if not os.path.exists(split_path):
            print(f"Warning: Split file not found: {split_path}")
            return []

        with open(split_path, 'r') as f:
            # IDs are like "train/AHTD3A0058_Para1_1" - extract just the ID
            ids = []
            for line in f:
                line = line.strip()
                if line:
                    # Remove split prefix (train/, val/, test/)
                    sample_id = line.split('/')[-1]
                    ids.append(sample_id)
        return ids

    def load_labels_from_txt(self, split_name):
        """Load labels from the split _text.txt file"""
        # Map split names to file names (use *_text.txt for proper labels)
        txt_files = {
            "train": "train_text.txt",
            "valid": "val_text.txt",
            "test": "test_text.txt"
        }

        txt_file = txt_files.get(split_name)
        if not txt_file:
            return {}

        txt_path = os.path.join(self.source_fold_path, txt_file)
        if not os.path.exists(txt_path):
            return {}

        labels = {}
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: split/sample_id label_text
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                sample_path = parts[0]
                label = parts[1]
                # Extract sample ID
                sample_id = sample_path.split('/')[-1]
                labels[sample_id] = label
        return labels

    def load_label(self, sample_id, split_labels=None):
        """Load transcription for a sample"""
        # First try from split_labels dict (preloaded from .txt file)
        if split_labels and sample_id in split_labels:
            return split_labels[sample_id]

        # Then try from labels/ directory
        label_path = os.path.join(self.source_fold_path, "labels", f"{sample_id}.txt")

        if not os.path.exists(label_path):
            return None

        try:
            with open(label_path, 'r', encoding=self.label_encoding) as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Warning: Could not read label {label_path}: {e}")
            return None

    def load_image(self, sample_id, split_name=None):
        """Load image for a sample"""
        # Try images/ directory first
        img_path = os.path.join(self.source_fold_path, "images", f"{sample_id}.jpg")

        if not os.path.exists(img_path):
            # Try data/{split}/ directory (for test set)
            if split_name:
                split_dir = "val" if split_name == "valid" else split_name
                img_path = os.path.join(self.source_fold_path, "data", split_dir, f"{sample_id}.jpg")

        if not os.path.exists(img_path):
            return None

        try:
            img = Image.open(img_path)
            return np.array(img)
        except Exception as e:
            print(f"Warning: Could not read image {img_path}: {e}")
            return None

    def format_khatt_line(self):
        """Format KHATT dataset at line level"""
        print("Formatting KHATT dataset at line level...")
        print(f"Label encoding: {self.label_encoding}")

        for set_name in self.set_names:
            sample_ids = self.load_split_ids(set_name)
            print(f"Processing {set_name}: {len(sample_ids)} samples")

            # Preload labels from .txt file (useful for test set)
            split_labels = self.load_labels_from_txt(set_name)
            if split_labels:
                print(f"  Loaded {len(split_labels)} labels from .txt file")

            processed = 0
            skipped = 0

            for i, sample_id in enumerate(sample_ids):
                # Load label
                text = self.load_label(sample_id, split_labels)
                if text is None or not text.strip():
                    skipped += 1
                    continue

                # Load image
                img = self.load_image(sample_id, set_name)
                if img is None or img.size == 0:
                    skipped += 1
                    continue

                # Resize to target DPI if needed
                if self.source_dpi != self.dpi:
                    img = self.resize(img, self.source_dpi, self.dpi)

                # Save image
                new_img_name = f"{set_name}_{processed}.png"
                new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)

                if len(img.shape) == 2:
                    Image.fromarray(img, mode='L').save(new_img_path)
                elif img.shape[2] == 3:
                    Image.fromarray(img, mode='RGB').save(new_img_path)
                else:
                    Image.fromarray(img).save(new_img_path)

                # Format and store label (plain string for CTC)
                text = self.format_text_label(text)
                self.charset = self.charset.union(set(text))
                self.gt[set_name][new_img_name] = text
                processed += 1

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(sample_ids)}...")

            print(f"  {set_name}: {processed} processed, {skipped} skipped")

    def get_paragraph_id(self, sample_id):
        """Extract paragraph ID from sample ID (e.g., AHTD3A0001_Para1_3 -> AHTD3A0001_Para1)"""
        parts = sample_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return sample_id

    def get_line_number(self, sample_id):
        """Extract line number from sample ID (e.g., AHTD3A0001_Para1_3 -> 3)"""
        parts = sample_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    def stack_images_vertically(self, images, padding=10, bg_color=255):
        """Stack multiple images vertically with padding between them"""
        if not images:
            return None

        # Get max width and total height
        max_width = max(img.shape[1] for img in images)
        total_height = sum(img.shape[0] for img in images) + padding * (len(images) - 1)

        # Determine if images are grayscale or RGB
        if len(images[0].shape) == 2:
            # Grayscale
            stacked = np.full((total_height, max_width), bg_color, dtype=np.uint8)
        else:
            # RGB
            stacked = np.full((total_height, max_width, images[0].shape[2]), bg_color, dtype=np.uint8)

        y_offset = 0
        for img in images:
            h, w = img.shape[:2]
            # Center horizontally
            x_offset = (max_width - w) // 2
            if len(img.shape) == 2:
                stacked[y_offset:y_offset + h, x_offset:x_offset + w] = img
            else:
                stacked[y_offset:y_offset + h, x_offset:x_offset + w, :] = img
            y_offset += h + padding

        return stacked

    def format_khatt_paragraph(self):
        """Format KHATT dataset at paragraph level by grouping lines"""
        print("Formatting KHATT dataset at PARAGRAPH level...")
        print(f"Label encoding: {self.label_encoding}")
        print("Grouping lines by paragraph ID...")

        for set_name in self.set_names:
            sample_ids = self.load_split_ids(set_name)
            print(f"\nProcessing {set_name}: {len(sample_ids)} line samples")

            # Preload labels from .txt file
            split_labels = self.load_labels_from_txt(set_name)
            if split_labels:
                print(f"  Loaded {len(split_labels)} labels from .txt file")

            # Group lines by paragraph ID
            paragraphs = defaultdict(list)
            for sample_id in sample_ids:
                para_id = self.get_paragraph_id(sample_id)
                line_num = self.get_line_number(sample_id)
                paragraphs[para_id].append((line_num, sample_id))

            print(f"  Found {len(paragraphs)} unique paragraphs")

            processed = 0
            skipped = 0

            for para_id, lines in paragraphs.items():
                # Sort lines by line number
                lines.sort(key=lambda x: x[0])

                # Load all line images and labels
                line_images = []
                line_texts = []
                valid = True

                for line_num, sample_id in lines:
                    # Load label
                    text = self.load_label(sample_id, split_labels)
                    if text is None or not text.strip():
                        valid = False
                        break

                    # Load image
                    img = self.load_image(sample_id, set_name)
                    if img is None or img.size == 0:
                        valid = False
                        break

                    # Resize to target DPI if needed
                    if self.source_dpi != self.dpi:
                        img = self.resize(img, self.source_dpi, self.dpi)

                    line_images.append(img)
                    line_texts.append(self.format_text_label(text))

                if not valid or not line_images:
                    skipped += 1
                    continue

                # Stack images vertically
                para_img = self.stack_images_vertically(line_images, padding=15)
                if para_img is None:
                    skipped += 1
                    continue

                # Join texts with newlines
                para_text = '\n'.join(line_texts)

                # Save paragraph image
                new_img_name = f"{set_name}_{processed}.png"
                new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)

                if len(para_img.shape) == 2:
                    Image.fromarray(para_img, mode='L').save(new_img_path)
                elif para_img.shape[2] == 3:
                    Image.fromarray(para_img, mode='RGB').save(new_img_path)
                else:
                    Image.fromarray(para_img).save(new_img_path)

                # Store paragraph label
                self.charset = self.charset.union(set(para_text))
                self.gt[set_name][new_img_name] = para_text
                processed += 1

                if processed % 200 == 0:
                    print(f"  Processed {processed} paragraphs...")

            print(f"  {set_name}: {processed} paragraphs, {skipped} skipped")

        # Print summary
        print(f"\nParagraph formatting complete!")
        print(f"Charset size: {len(self.charset)}")


def format_khatt_paragraph(raw_data_path, output_path, dpi=150):
    """
    Convenience function to format KHATT dataset at paragraph level.

    Args:
        raw_data_path: Path to raw KHATT data directory
        output_path: Path for formatted output
        dpi: Target DPI for output images
    """
    formatter = KHATTDatasetFormatter(
        level="paragraph",
        dpi=dpi,
        raw_data_path=raw_data_path
    )
    formatter.target_fold_path = output_path
    formatter.format()
    return formatter


def format_khatt_dataset(raw_data_path, output_path, dpi=150):
    """
    Convenience function to format KHATT dataset.

    Args:
        raw_data_path: Path to raw KHATT data directory
        output_path: Path for formatted output
        dpi: Target DPI for output images
    """
    formatter = KHATTDatasetFormatter(
        level="line",
        dpi=dpi,
        raw_data_path=raw_data_path
    )
    formatter.target_fold_path = output_path
    formatter.format()
    return formatter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format KHATT dataset for HAND framework")
    parser.add_argument("--raw_path", type=str, required=True,
                        help="Path to raw KHATT data")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path for formatted output")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Target DPI for output images")

    args = parser.parse_args()

    format_khatt_dataset(
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        dpi=args.dpi
    )
