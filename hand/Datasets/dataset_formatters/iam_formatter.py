#  IAM Handwriting Database Formatter for HAND framework
#  Formats IAM dataset for OCR/HTR training at line and paragraph levels

from hand.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class IAMDatasetFormatter(OCRDatasetFormatter):
    """
    Formatter for the IAM Handwriting Database.

    Supports TWO common distribution formats:

    Format 1 - Pre-extracted lines (preferred, faster):
    IAM/
    ├── lines/          # Pre-extracted line images
    │   └── {form_prefix}/
    │       └── {form_id}/
    │           └── {line_id}.png
    ├── ascii/          # Text transcriptions (optional, uses xml if not present)
    │   └── lines.txt   # Format: line_id ok/err gray_level #components transcript
    └── partitions/
        ├── trainset.txt
        ├── testset.txt
        └── validationset1.txt

    Format 2 - Form images (requires cropping):
    IAM/
    ├── data/           # Form images organized by writer ID or form prefix
    │   └── {writer_id}/
    │       └── {form_id}.png
    ├── xml/            # XML annotations with transcriptions and bounding boxes
    │   └── {form_id}.xml
    └── partitions/
        ├── trainset.txt
        ├── testset.txt
        └── validationset1.txt
    """

    def __init__(self, level, set_names=["train", "valid", "test"], dpi=150,
                 source_dpi=300, raw_data_path=None):
        """
        Args:
            level: 'line' or 'paragraph'
            set_names: dataset splits to create
            dpi: target DPI for output images
            source_dpi: source DPI of IAM images (300 dpi)
            raw_data_path: path to raw IAM data (overrides default)
        """
        super(IAMDatasetFormatter, self).__init__("IAM", level, "", set_names)

        self.dpi = dpi
        self.source_dpi = source_dpi

        # Override source path if provided
        if raw_data_path:
            self.source_fold_path = raw_data_path

        # Map partition files to set names
        self.partition_files = {
            "train": "trainset.txt",
            "valid": "validationset1.txt",  # Using validationset1 as validation
            "test": "testset.txt"
        }

        self.map_datasets_files.update({
            "IAM": {
                "line": {
                    "arx_files": [],  # No archive files, data is already extracted
                    "needed_files": [],
                    "format_function": self.format_iam_line,
                },
                "paragraph": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_iam_paragraph,
                },
                "page": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_iam_page,
                },
            }
        })

    def init_format(self):
        """Initialize formatting - create output directories"""
        os.makedirs(self.target_fold_path, exist_ok=True)
        for set_name in self.set_names:
            os.makedirs(os.path.join(self.target_fold_path, set_name), exist_ok=True)

    def end_format(self):
        """Save labels.pkl file (no temp folder cleanup for IAM)"""
        import pickle
        with open(os.path.join(self.target_fold_path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": self.gt,
                "charset": sorted(list(self.charset)),
            }, f)
        print(f"Saved labels.pkl with {len(self.charset)} characters")
        for set_name in self.set_names:
            print(f"  {set_name}: {len(self.gt[set_name])} samples")

    def parse_line_id(self, line_id):
        """
        Parse IAM line ID to extract form_id and line components.

        Example: "a01-000u-00" -> form_id="a01-000u", line_num="00"
        """
        parts = line_id.rsplit('-', 1)
        form_id = parts[0]  # e.g., "a01-000u"
        line_num = parts[1] if len(parts) > 1 else "00"
        return form_id, line_num

    def get_writer_folder(self, xml_path, form_id):
        """
        Get writer folder from XML or derive from form naming.
        Form ID format: {letter}{writer_id}-{form_num}
        e.g., "a01-000u" -> writer folder could be "000" or similar
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            writer_id = root.get('writer-id')
            if writer_id:
                return writer_id.zfill(3)  # Pad to 3 digits
        except:
            pass

        # Fallback: derive from form_id structure
        # Format: a01-000u -> extract writer/folder info
        parts = form_id.split('-')
        if len(parts) >= 1:
            # First part is like "a01", "b01", etc.
            return parts[0]
        return None

    def load_partition(self, partition_file):
        """Load line IDs from partition file"""
        partition_path = os.path.join(self.source_fold_path, "partitions", partition_file)
        if not os.path.exists(partition_path):
            # Try alternative path
            partition_path = os.path.join(self.source_fold_path, partition_file)

        if not os.path.exists(partition_path):
            print(f"Warning: Partition file not found: {partition_path}")
            return []

        with open(partition_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def load_xml_data(self, form_id):
        """
        Load XML annotation for a form and extract all line data.
        Returns dict of line_id -> {text, coords}
        """
        xml_path = os.path.join(self.source_fold_path, "xml", f"{form_id}.xml")

        if not os.path.exists(xml_path):
            return None, None

        tree = ET.parse(xml_path)
        root = tree.getroot()
        writer_id = root.get('writer-id', '000')

        lines_data = {}

        # Find all line elements
        for line in root.iter('line'):
            line_id = line.get('id')
            text = line.get('text', '')

            # Decode HTML entities that might be in the text
            text = text.replace('&quot;', '"').replace('&amp;', '&')
            text = text.replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&apos;', "'")

            # Skip lines with segmentation errors if needed
            seg_status = line.get('segmentation', 'ok')

            # Get bounding box from component (cmp) elements
            # IAM XML structure: <line> -> <word> -> <cmp x="" y="" width="" height="" />
            try:
                # Find all component elements within the line
                cmps = line.findall('.//cmp')
                if cmps:
                    coords = []
                    for cmp in cmps:
                        x = int(cmp.get('x', 0))
                        y = int(cmp.get('y', 0))
                        w = int(cmp.get('width', 0))
                        h = int(cmp.get('height', 0))
                        if w > 0 and h > 0:  # Skip invalid components
                            coords.append((x, y, x + w, y + h))

                    if coords:
                        left = min(c[0] for c in coords)
                        top = min(c[1] for c in coords)
                        right = max(c[2] for c in coords)
                        bottom = max(c[3] for c in coords)
                    else:
                        print(f"Warning: No valid cmp coords for line {line_id}")
                        continue
                else:
                    # Fallback: try word-level coordinates
                    words = line.findall('.//word')
                    if words:
                        coords = []
                        for word in words:
                            x = int(word.get('x', 0))
                            y = int(word.get('y', 0))
                            w = int(word.get('width', 0))
                            h = int(word.get('height', 0))
                            if w > 0 and h > 0:
                                coords.append((x, y, x + w, y + h))

                        if coords:
                            left = min(c[0] for c in coords)
                            top = min(c[1] for c in coords)
                            right = max(c[2] for c in coords)
                            bottom = max(c[3] for c in coords)
                        else:
                            print(f"Warning: No valid word coords for line {line_id}")
                            continue
                    else:
                        print(f"Warning: No cmp or word elements for line {line_id}")
                        continue

                lines_data[line_id] = {
                    'text': text,
                    'coords': {
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom
                    },
                    'segmentation': seg_status
                }
            except Exception as e:
                print(f"Warning: Could not parse coordinates for line {line_id}: {e}")
                continue

        return lines_data, writer_id

    def find_form_image(self, form_id, writer_id=None):
        """
        Find the form image file.
        Tries multiple possible locations based on IAM structure.
        """
        possible_paths = []

        # Extract form prefix for folder structure
        # Form ID like "a01-000u" -> folder could be "a01" or writer_id
        form_prefix = form_id.split('-')[0]

        if writer_id:
            possible_paths.append(os.path.join(self.source_fold_path, "data", writer_id, f"{form_id}.png"))
            possible_paths.append(os.path.join(self.source_fold_path, "data", writer_id.zfill(3), f"{form_id}.png"))

        possible_paths.extend([
            os.path.join(self.source_fold_path, "data", form_prefix, f"{form_id}.png"),
            os.path.join(self.source_fold_path, "forms", f"{form_id}.png"),
            os.path.join(self.source_fold_path, "data", f"{form_id}.png"),
            os.path.join(self.source_fold_path, f"{form_id}.png"),
        ])

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def load_lines_txt(self):
        """
        Load transcriptions from ascii/lines.txt file (common IAM distribution format).
        Returns dict: line_id -> text
        """
        lines_txt_path = os.path.join(self.source_fold_path, "ascii", "lines.txt")
        if not os.path.exists(lines_txt_path):
            # Try alternative location
            lines_txt_path = os.path.join(self.source_fold_path, "lines.txt")

        if not os.path.exists(lines_txt_path):
            return None

        transcriptions = {}
        with open(lines_txt_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Format: line_id ok/err gray_level #components transcript
                # Example: a01-000u-00 ok 154 19 A|MOVE|to|stop|...
                parts = line.split(' ', 8)  # Split into max 9 parts
                if len(parts) < 9:
                    continue

                line_id = parts[0]
                seg_status = parts[1]
                # parts[2] = gray level
                # parts[3:8] = bounding box info
                text = parts[8]

                # Convert pipe-separated words to text
                text = text.replace('|', ' ')

                transcriptions[line_id] = {
                    'text': text,
                    'status': seg_status
                }

        print(f"Loaded {len(transcriptions)} transcriptions from lines.txt")
        return transcriptions

    def find_line_image(self, line_id):
        """
        Find pre-extracted line image.
        Line ID format: a01-000u-00
        Path structure: lines/a01/a01-000u/a01-000u-00.png
        """
        parts = line_id.split('-')
        if len(parts) < 3:
            return None

        form_prefix = parts[0]  # e.g., "a01"
        form_id = f"{parts[0]}-{parts[1]}"  # e.g., "a01-000u"

        possible_paths = [
            os.path.join(self.source_fold_path, "lines", form_prefix, form_id, f"{line_id}.png"),
            os.path.join(self.source_fold_path, "lines", form_prefix, f"{line_id}.png"),
            os.path.join(self.source_fold_path, "lines", f"{line_id}.png"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def format_iam_line_preextracted(self, transcriptions):
        """Format IAM using pre-extracted line images"""
        print("Using pre-extracted line images...")

        for set_name in self.set_names:
            partition_file = self.partition_files.get(set_name)
            if not partition_file:
                print(f"No partition file for {set_name}")
                continue

            line_ids = self.load_partition(partition_file)
            print(f"Processing {set_name}: {len(line_ids)} lines")

            processed = 0
            skipped = 0

            for i, line_id in enumerate(line_ids):
                # Get transcription
                if line_id not in transcriptions:
                    skipped += 1
                    continue

                trans = transcriptions[line_id]
                text = trans['text'].strip()

                if not text:
                    skipped += 1
                    continue

                # Find line image
                img_path = self.find_line_image(line_id)
                if img_path is None:
                    skipped += 1
                    continue

                try:
                    # Load image
                    line_img = np.array(Image.open(img_path))

                    if line_img.size == 0:
                        skipped += 1
                        continue

                    # Resize to target DPI if needed
                    if self.source_dpi != self.dpi:
                        line_img = self.resize(line_img, self.source_dpi, self.dpi)

                    # Save line image
                    new_img_name = f"{set_name}_{processed}.png"
                    new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)

                    if len(line_img.shape) == 2:
                        Image.fromarray(line_img, mode='L').save(new_img_path)
                    else:
                        Image.fromarray(line_img).save(new_img_path)

                    # Update ground truth - store as plain string for CTC training
                    text = self.format_text_label(text)
                    self.charset = self.charset.union(set(text))
                    self.gt[set_name][new_img_name] = text
                    processed += 1

                except Exception as e:
                    print(f"Error processing line {line_id}: {e}")
                    skipped += 1
                    continue

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(line_ids)} lines...")

            print(f"  {set_name}: {processed} processed, {skipped} skipped")

    def format_iam_line(self):
        """Format IAM dataset at line level"""
        print("Formatting IAM dataset at line level...")

        # Check if pre-extracted lines exist
        lines_dir = os.path.join(self.source_fold_path, "lines")
        if os.path.exists(lines_dir):
            # Try to load transcriptions from lines.txt
            transcriptions = self.load_lines_txt()
            if transcriptions:
                return self.format_iam_line_preextracted(transcriptions)
            else:
                # Try loading from XML if lines.txt not available
                print("lines.txt not found, will try XML annotations...")

        print("Using form images with XML annotations (cropping required)...")

        # Cache for loaded form images
        form_cache = {}
        xml_cache = {}

        for set_name in self.set_names:
            partition_file = self.partition_files.get(set_name)
            if not partition_file:
                print(f"No partition file for {set_name}")
                continue

            line_ids = self.load_partition(partition_file)
            print(f"Processing {set_name}: {len(line_ids)} lines")

            processed = 0
            skipped = 0

            for i, line_id in enumerate(line_ids):
                form_id, line_num = self.parse_line_id(line_id)

                # Load XML data (cached)
                if form_id not in xml_cache:
                    xml_data, writer_id = self.load_xml_data(form_id)
                    xml_cache[form_id] = (xml_data, writer_id)
                else:
                    xml_data, writer_id = xml_cache[form_id]

                if xml_data is None or line_id not in xml_data:
                    skipped += 1
                    continue

                line_data = xml_data[line_id]

                # Skip empty text
                if not line_data['text'].strip():
                    skipped += 1
                    continue

                # Load form image (cached)
                if form_id not in form_cache:
                    img_path = self.find_form_image(form_id, writer_id)
                    if img_path:
                        form_cache[form_id] = np.array(Image.open(img_path))
                    else:
                        form_cache[form_id] = None
                        print(f"Warning: Image not found for form {form_id}")

                form_img = form_cache.get(form_id)
                if form_img is None:
                    skipped += 1
                    continue

                # Crop line image
                coords = line_data['coords']
                try:
                    # Add small padding
                    pad = 5
                    top = max(0, coords['top'] - pad)
                    bottom = min(form_img.shape[0], coords['bottom'] + pad)
                    left = max(0, coords['left'] - pad)
                    right = min(form_img.shape[1], coords['right'] + pad)

                    line_img = form_img[top:bottom, left:right].copy()

                    if line_img.size == 0:
                        skipped += 1
                        continue

                    # Resize to target DPI
                    if self.source_dpi != self.dpi:
                        line_img = self.resize(line_img, self.source_dpi, self.dpi)

                    # Save line image
                    new_img_name = f"{set_name}_{processed}.png"
                    new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)

                    if len(line_img.shape) == 2:
                        # Grayscale
                        Image.fromarray(line_img, mode='L').save(new_img_path)
                    else:
                        Image.fromarray(line_img).save(new_img_path)

                    # Update ground truth - store as plain string for CTC training
                    text = self.format_text_label(line_data['text'])
                    self.charset = self.charset.union(set(text))
                    self.gt[set_name][new_img_name] = text
                    processed += 1

                except Exception as e:
                    print(f"Error processing line {line_id}: {e}")
                    skipped += 1
                    continue

                # Progress update
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(line_ids)} lines...")

            print(f"  {set_name}: {processed} processed, {skipped} skipped")

            # Clear caches between sets to save memory
            form_cache.clear()

    def format_iam_paragraph(self):
        """Format IAM dataset at paragraph level (grouped by form text regions)"""
        print("Formatting IAM dataset at paragraph level...")
        print("Note: IAM paragraph-level grouping based on form structure")

        # For IAM, paragraphs could be:
        # 1. All lines in a form concatenated (simpler)
        # 2. Detected text blocks based on spacing

        # Here we'll group lines by form as "paragraphs"
        form_lines = {}  # form_id -> {set_name, lines_data}

        for set_name in self.set_names:
            partition_file = self.partition_files.get(set_name)
            if not partition_file:
                continue

            line_ids = self.load_partition(partition_file)

            for line_id in line_ids:
                form_id, _ = self.parse_line_id(line_id)

                if form_id not in form_lines:
                    form_lines[form_id] = {'set_name': set_name, 'line_ids': []}
                form_lines[form_id]['line_ids'].append(line_id)

        print(f"Found {len(form_lines)} forms to process as paragraphs")

        xml_cache = {}
        processed_counts = {s: 0 for s in self.set_names}

        for form_id, form_info in form_lines.items():
            set_name = form_info['set_name']
            line_ids = form_info['line_ids']

            # Load XML data
            if form_id not in xml_cache:
                xml_data, writer_id = self.load_xml_data(form_id)
                xml_cache[form_id] = (xml_data, writer_id)
            else:
                xml_data, writer_id = xml_cache[form_id]

            if xml_data is None:
                continue

            # Find form image
            img_path = self.find_form_image(form_id, writer_id)
            if not img_path:
                continue

            form_img = np.array(Image.open(img_path))

            # Collect all lines for this form
            lines = []
            for line_id in sorted(line_ids):
                if line_id in xml_data and xml_data[line_id]['text'].strip():
                    lines.append({
                        'id': line_id,
                        **xml_data[line_id]
                    })

            if not lines:
                continue

            # Calculate paragraph bounding box
            para_top = min(l['coords']['top'] for l in lines) - 10
            para_bottom = max(l['coords']['bottom'] for l in lines) + 10
            para_left = min(l['coords']['left'] for l in lines) - 10
            para_right = max(l['coords']['right'] for l in lines) + 10

            # Clamp to image bounds
            para_top = max(0, para_top)
            para_bottom = min(form_img.shape[0], para_bottom)
            para_left = max(0, para_left)
            para_right = min(form_img.shape[1], para_right)

            # Crop paragraph image
            para_img = form_img[para_top:para_bottom, para_left:para_right].copy()

            if para_img.size == 0:
                continue

            # Resize to target DPI
            if self.source_dpi != self.dpi:
                para_img = self.resize(para_img, self.source_dpi, self.dpi)

            # Save paragraph image
            idx = processed_counts[set_name]
            new_img_name = f"{set_name}_{idx}.png"
            new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)

            if len(para_img.shape) == 2:
                Image.fromarray(para_img, mode='L').save(new_img_path)
            else:
                Image.fromarray(para_img).save(new_img_path)

            # Combine line texts into paragraph text
            para_text = '\n'.join(self.format_text_label(l['text']) for l in lines)
            self.charset = self.charset.union(set(para_text))

            # Calculate relative line coordinates
            dpi_ratio = self.dpi / self.source_dpi
            line_info = []
            for l in lines:
                line_info.append({
                    'text': self.format_text_label(l['text']),
                    'top': int((l['coords']['top'] - para_top) * dpi_ratio),
                    'bottom': int((l['coords']['bottom'] - para_top) * dpi_ratio),
                    'left': int((l['coords']['left'] - para_left) * dpi_ratio),
                    'right': int((l['coords']['right'] - para_left) * dpi_ratio),
                })

            self.gt[set_name][new_img_name] = {
                "text": para_text,
                "lines": line_info,
                "nb_cols": 1,
            }
            processed_counts[set_name] += 1

        for set_name in self.set_names:
            print(f"  {set_name}: {processed_counts[set_name]} paragraphs")

    def format_iam_page(self):
        """
        Format IAM dataset at page/form level.
        Each form image is used as a full page with all its line transcriptions.
        This is suitable for full-page HTR training similar to READ_2016 page level.
        """
        print("Formatting IAM dataset at page/form level...")

        # Group lines by form
        form_lines = {}  # form_id -> {set_name, line_ids}

        for set_name in self.set_names:
            partition_file = self.partition_files.get(set_name)
            if not partition_file:
                continue

            line_ids = self.load_partition(partition_file)

            for line_id in line_ids:
                form_id, _ = self.parse_line_id(line_id)

                if form_id not in form_lines:
                    form_lines[form_id] = {'set_name': set_name, 'line_ids': []}
                form_lines[form_id]['line_ids'].append(line_id)

        print(f"Found {len(form_lines)} forms to process as pages")

        xml_cache = {}
        processed_counts = {s: 0 for s in self.set_names}

        for form_id, form_info in form_lines.items():
            set_name = form_info['set_name']
            line_ids = form_info['line_ids']

            # Load XML data
            if form_id not in xml_cache:
                xml_data, writer_id = self.load_xml_data(form_id)
                xml_cache[form_id] = (xml_data, writer_id)
            else:
                xml_data, writer_id = xml_cache[form_id]

            if xml_data is None:
                continue

            # Find form image
            img_path = self.find_form_image(form_id, writer_id)
            if not img_path:
                continue

            form_img = np.array(Image.open(img_path))

            # Collect all lines for this form (sorted by line ID for reading order)
            lines = []
            for line_id in sorted(line_ids):
                if line_id in xml_data and xml_data[line_id]['text'].strip():
                    lines.append({
                        'id': line_id,
                        **xml_data[line_id]
                    })

            if not lines:
                continue

            # Resize full form image to target DPI
            if self.source_dpi != self.dpi:
                form_img = self.resize(form_img, self.source_dpi, self.dpi)

            # Save page/form image
            idx = processed_counts[set_name]
            new_img_name = f"{set_name}_{idx}.png"
            new_img_path = os.path.join(self.target_fold_path, set_name, new_img_name)

            if len(form_img.shape) == 2:
                Image.fromarray(form_img, mode='L').save(new_img_path)
            else:
                Image.fromarray(form_img).save(new_img_path)

            # Combine line texts into page text (newline separated)
            page_text = '\n'.join(self.format_text_label(l['text']) for l in lines)
            self.charset = self.charset.union(set(page_text))

            # Calculate line coordinates adjusted for DPI
            dpi_ratio = self.dpi / self.source_dpi
            line_info = []
            for l in lines:
                line_info.append({
                    'text': self.format_text_label(l['text']),
                    'top': int(l['coords']['top'] * dpi_ratio),
                    'bottom': int(l['coords']['bottom'] * dpi_ratio),
                    'left': int(l['coords']['left'] * dpi_ratio),
                    'right': int(l['coords']['right'] * dpi_ratio),
                })

            # Build paragraphs structure (single paragraph per form for IAM)
            paragraphs = [{
                'label': page_text,
                'lines': line_info,
                'mode': 'body',
                'top': min(l['top'] for l in line_info) if line_info else 0,
                'bottom': max(l['bottom'] for l in line_info) if line_info else 0,
                'left': min(l['left'] for l in line_info) if line_info else 0,
                'right': max(l['right'] for l in line_info) if line_info else 0,
            }]

            # Page-level ground truth structure (compatible with READ_2016 page format)
            page_width = int(form_img.shape[1])
            page_label = {
                'text': page_text,
                'paragraphs': paragraphs,
                'nb_cols': 1,
                'side': 'single',
                'top': paragraphs[0]['top'],
                'bottom': paragraphs[0]['bottom'],
                'left': paragraphs[0]['left'],
                'right': paragraphs[0]['right'],
                'page_width': page_width,
            }

            self.gt[set_name][new_img_name] = {
                "text": page_text,
                "nb_cols": 1,
                "pages": [page_label],
            }
            processed_counts[set_name] += 1

        for set_name in self.set_names:
            print(f"  {set_name}: {processed_counts[set_name]} pages/forms")


def format_iam_dataset(raw_data_path, output_path, level="line", dpi=150):
    """
    Convenience function to format IAM dataset.

    Args:
        raw_data_path: Path to raw IAM data directory
        output_path: Path for formatted output
        level: 'line', 'paragraph', or 'page'
        dpi: Target DPI for output images
    """
    formatter = IAMDatasetFormatter(
        level=level,
        dpi=dpi,
        raw_data_path=raw_data_path
    )
    formatter.target_fold_path = output_path
    formatter.format()
    return formatter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format IAM dataset for HAND framework")
    parser.add_argument("--raw_path", type=str, required=True,
                        help="Path to raw IAM data")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path for formatted output")
    parser.add_argument("--level", type=str, default="line",
                        choices=["line", "paragraph", "page"],
                        help="Formatting level (line, paragraph, or page)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Target DPI for output images")

    args = parser.parse_args()

    format_iam_dataset(
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        level=args.level,
        dpi=args.dpi
    )
