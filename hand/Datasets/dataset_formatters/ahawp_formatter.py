#  AHAWP Arabic Handwriting Dataset Formatter for HAND framework
#  Arabic Handwritten Alphabets, Words and Paragraphs Per User (AHAWP)
#  Supports curriculum learning: character → word → paragraph levels

from hand.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
import os
import numpy as np
from PIL import Image
import random


class AHAWPDatasetFormatter(OCRDatasetFormatter):
    """
    Formatter for the AHAWP (Arabic Handwritten Alphabets, Words and Paragraphs) Dataset.

    AHAWP contains:
    - 82 users
    - 65 character types (Arabic letters in different positions)
    - 10 unique Arabic words
    - 3 paragraphs per user (fixed text)

    Total samples:
    - Characters: ~53,199 samples
    - Words: ~8,144 samples
    - Paragraphs: ~241 samples

    Curriculum Learning Levels:
    1. character: Isolated Arabic characters (CTC training)
    2. word: Isolated Arabic words (CTC training)
    3. paragraph: Full paragraphs (seq2seq HAND training)
    """

    # Arabic character mappings (romanized name → Arabic character)
    CHAR_MAPPING = {
        # Alif variants
        'alif_regular': 'ا',
        'alif_end': 'ـا',
        'alif_hamza': 'أ',

        # Beh (ب)
        'beh_regular': 'ب',
        'beh_begin': 'بـ',
        'beh_middle': 'ـبـ',
        'beh_end': 'ـب',

        # Jeem (ج) - also covers ح خ in some forms
        'jeem_regular': 'ج',
        'jeem_begin': 'جـ',
        'jeem_middle': 'ـجـ',
        'jeem_end': 'ـج',

        # Dal (د)
        'dal_regular': 'د',
        'dal_end': 'ـد',

        # Raa (ر)
        'raa_regular': 'ر',
        'raa_end': 'ـر',

        # Seen (س)
        'seen_regular': 'س',
        'seen_begin': 'سـ',
        'seen_middle': 'ـسـ',
        'seen_end': 'ـس',

        # Sad (ص)
        'sad_regular': 'ص',
        'sad_begin': 'صـ',
        'sad_middle': 'ـصـ',
        'sad_end': 'ـص',

        # Tah (ط)
        'tah_regular': 'ط',
        'tah_middle': 'ـطـ',
        'tah_end': 'ـط',

        # Ain (ع)
        'ain_regular': 'ع',
        'ain_begin': 'عـ',
        'ain_middle': 'ـعـ',
        'ain_end': 'ـع',

        # Feh (ف)
        'feh_regular': 'ف',
        'feh_begin': 'فـ',
        'feh_middle': 'ـفـ',
        'feh_end': 'ـف',

        # Qaf (ق)
        'qaf_regular': 'ق',
        'qaf_begin': 'قـ',
        'qaf_middle': 'ـقـ',
        'qaf_end': 'ـق',

        # Kaf (ك)
        'kaf_regular': 'ك',
        'kaf_begin': 'كـ',
        'kaf_middle': 'ـكـ',
        'kaf_end': 'ـك',

        # Lam (ل)
        'lam_regular': 'ل',
        'lam_begin': 'لـ',
        'lam_middle': 'ـلـ',
        'lam_end': 'ـل',
        'lam_alif': 'لا',

        # Meem (م)
        'meem_regular': 'م',
        'meem_begin': 'مـ',
        'meem_middle': 'ـمـ',
        'meem_end': 'ـم',

        # Noon (ن)
        'noon_regular': 'ن',
        'noon_begin': 'نـ',
        'noon_middle': 'ـنـ',
        'noon_end': 'ـن',

        # Heh (ه)
        'heh_regular': 'ه',
        'heh_begin': 'هـ',
        'heh_middle': 'ـهـ',
        'heh_end': 'ـه',

        # Waw (و)
        'waw_regular': 'و',
        'waw_end': 'ـو',

        # Yaa (ي)
        'yaa_regular': 'ي',
        'yaa_begin': 'يـ',
        'yaa_middle': 'ـيـ',
        'yaa_end': 'ـي',
    }

    # Arabic word mappings (romanized → Arabic)
    WORD_MAPPING = {
        'azan': 'أذان',           # Call to prayer
        'sakhar': 'صخر',          # Rock
        'mustadhafeen': 'مستضعفين',  # The oppressed
        'abjadiyah': 'أبجدية',     # Alphabet
        'fasayakfeekahum': 'فسيكفيكهم',  # Quranic verse
        'ghazaal': 'غزال',        # Gazelle
        'ghaleez': 'غليظ',        # Thick/harsh
        'qashtah': 'قشطة',        # Cream
        'shateerah': 'شطيرة',     # Sandwich
        'mehras': 'مهراس',        # Mortar
    }

    # Paragraph transcriptions (fixed text written by all users)
    # Verified from actual paragraph images
    # NOTE: Line breaks (\n) added at sentence boundaries for HAND training
    PARAGRAPH_MAPPING = {
        # Paragraph 1: Article about climate change (5 lines)
        'paragraph_1': 'هذا مقال عن تغير المناخ.\nيمكن أن يكون سبب تغير المناخ في العالم بسبب الأنشطة المختلفة.\nعندما يحدث تغير المناخ.\nدرجات الحرارة يمكن أن تزيد بشكل كبير.\nخلال القرن الماضي، أطلقت الأنشطة البشرية كميات كبيرة من ثاني أكسيد الكربون وغازات الدفيئة التي نرى في الغلاف الجوي.',
        # Paragraph 2: Career/curiosity advice (4 lines)
        'paragraph_2': 'بدلاً من اتخاذ قرارات محددة بشأن المسار الوظيفي أعتقد أنه يجب عليك أن تكون فضولياً.\nأحصل على فضول حول الطريقة التي يعمل بها العالم.\nلاحظ أصحابك الخاصة وأحبب شيء ما وقت صغيرة يمكنك من خلالها ممارسة الشغف في شيء ما.\nحتى إذا لم تتمكن من العثور على طريقة لكسب المال منه في الذات.',
        # Paragraph 3: Stephen Hawking quote about stars (4 lines)
        'paragraph_3': 'أنظر إلى النجوم وليس إلى أسفل عند قدميك.\nحاول أن تفهم ما تراه.\nوتساءل عما يجعل الكون موجوداً.\nكن فضولياً.',
    }

    def __init__(self, level="character", set_names=["train", "valid", "test"],
                 dpi=150, raw_data_path=None, train_ratio=0.7, valid_ratio=0.15):
        """
        Args:
            level: 'character', 'word', or 'paragraph'
            set_names: dataset splits to create
            dpi: target DPI for output images
            raw_data_path: path to extracted AHAWP data
            train_ratio: ratio for training split
            valid_ratio: ratio for validation split (test = 1 - train - valid)
        """
        super(AHAWPDatasetFormatter, self).__init__("AHAWP", level, "", set_names)

        self.dpi = dpi
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = 1.0 - train_ratio - valid_ratio

        if raw_data_path:
            self.source_fold_path = raw_data_path

        self.map_datasets_files.update({
            "AHAWP": {
                "character": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_character_level,
                },
                "word": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_word_level,
                },
                "paragraph": {
                    "arx_files": [],
                    "needed_files": [],
                    "format_function": self.format_paragraph_level,
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
            if set_name in self.gt:
                print(f"  {set_name}: {len(self.gt[set_name])} samples")

    def get_split(self, index, total):
        """Determine split based on index"""
        train_end = int(total * self.train_ratio)
        valid_end = train_end + int(total * self.valid_ratio)

        if index < train_end:
            return "train"
        elif index < valid_end:
            return "valid"
        else:
            return "test"

    def format_character_level(self):
        """Format AHAWP dataset at character level"""
        print("Formatting AHAWP dataset at CHARACTER level...")
        print(f"Character types: {len(self.CHAR_MAPPING)}")

        # Path to isolated alphabets
        char_dir = os.path.join(self.source_fold_path, "isolated_alphabets_per_alphabet")

        if not os.path.exists(char_dir):
            print(f"ERROR: Character directory not found: {char_dir}")
            return

        # Collect all samples
        all_samples = []

        for char_folder in os.listdir(char_dir):
            char_path = os.path.join(char_dir, char_folder)
            if not os.path.isdir(char_path):
                continue

            # Get Arabic label
            arabic_char = self.CHAR_MAPPING.get(char_folder)
            if arabic_char is None:
                print(f"  Warning: No mapping for {char_folder}")
                continue

            for img_file in os.listdir(char_path):
                if img_file.endswith('.png'):
                    all_samples.append({
                        'img_path': os.path.join(char_path, img_file),
                        'label': arabic_char,
                        'char_type': char_folder
                    })

        print(f"Total character samples: {len(all_samples)}")

        # Shuffle and split
        random.seed(42)
        random.shuffle(all_samples)

        counts = {"train": 0, "valid": 0, "test": 0}

        for i, sample in enumerate(all_samples):
            split = self.get_split(i, len(all_samples))

            # Load and save image
            img = Image.open(sample['img_path'])
            img_array = np.array(img)

            new_img_name = f"{split}_{counts[split]}.png"
            new_img_path = os.path.join(self.target_fold_path, split, new_img_name)

            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img = Image.fromarray(img_array, mode='L').convert('RGB')
            img.save(new_img_path)

            # Store label
            text = sample['label']
            self.charset = self.charset.union(set(text))
            self.gt[split][new_img_name] = text
            counts[split] += 1

            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{len(all_samples)}...")

        print(f"Character level formatting complete:")
        for split, count in counts.items():
            print(f"  {split}: {count} samples")

    def format_word_level(self):
        """Format AHAWP dataset at word level"""
        print("Formatting AHAWP dataset at WORD level...")
        print(f"Word types: {len(self.WORD_MAPPING)}")

        # Path to isolated words
        word_dir = os.path.join(self.source_fold_path, "isolated_words_per_user")

        if not os.path.exists(word_dir):
            print(f"ERROR: Word directory not found: {word_dir}")
            return

        # Collect all samples
        all_samples = []

        for user_folder in os.listdir(word_dir):
            user_path = os.path.join(word_dir, user_folder)
            if not os.path.isdir(user_path):
                continue

            for img_file in os.listdir(user_path):
                if not img_file.endswith('.png'):
                    continue

                # Parse filename: user001_wordname_001.png
                parts = img_file.replace('.png', '').split('_')
                if len(parts) >= 2:
                    word_name = parts[1]
                    arabic_word = self.WORD_MAPPING.get(word_name)

                    if arabic_word is None:
                        print(f"  Warning: No mapping for word {word_name}")
                        continue

                    all_samples.append({
                        'img_path': os.path.join(user_path, img_file),
                        'label': arabic_word,
                        'word_type': word_name,
                        'user': user_folder
                    })

        print(f"Total word samples: {len(all_samples)}")

        # Shuffle and split
        random.seed(42)
        random.shuffle(all_samples)

        counts = {"train": 0, "valid": 0, "test": 0}

        for i, sample in enumerate(all_samples):
            split = self.get_split(i, len(all_samples))

            # Load and save image
            img = Image.open(sample['img_path'])
            img_array = np.array(img)

            new_img_name = f"{split}_{counts[split]}.png"
            new_img_path = os.path.join(self.target_fold_path, split, new_img_name)

            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img = Image.fromarray(img_array, mode='L').convert('RGB')
            img.save(new_img_path)

            # Store label
            text = sample['label']
            self.charset = self.charset.union(set(text))
            self.gt[split][new_img_name] = text
            counts[split] += 1

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(all_samples)}...")

        print(f"Word level formatting complete:")
        for split, count in counts.items():
            print(f"  {split}: {count} samples")

    def format_paragraph_level(self):
        """Format AHAWP dataset at paragraph level"""
        print("Formatting AHAWP dataset at PARAGRAPH level...")
        print(f"Paragraph types: {len(self.PARAGRAPH_MAPPING)}")
        print("NOTE: Paragraph transcriptions are estimated - verify against original source")

        # Path to paragraphs
        para_dir = os.path.join(self.source_fold_path, "paragraphs_per_user")

        if not os.path.exists(para_dir):
            print(f"ERROR: Paragraph directory not found: {para_dir}")
            return

        # Collect all samples
        all_samples = []

        for user_folder in os.listdir(para_dir):
            user_path = os.path.join(para_dir, user_folder)
            if not os.path.isdir(user_path):
                continue

            for img_file in os.listdir(user_path):
                if not img_file.endswith('.png'):
                    continue

                # Parse filename: user001_paragraph_1_001.png
                parts = img_file.replace('.png', '').split('_')
                if len(parts) >= 3 and parts[1] == 'paragraph':
                    para_num = parts[2]
                    para_name = f"paragraph_{para_num}"
                    arabic_text = self.PARAGRAPH_MAPPING.get(para_name)

                    if arabic_text is None:
                        print(f"  Warning: No mapping for {para_name}")
                        continue

                    all_samples.append({
                        'img_path': os.path.join(user_path, img_file),
                        'label': arabic_text,
                        'para_type': para_name,
                        'user': user_folder
                    })

        print(f"Total paragraph samples: {len(all_samples)}")

        # Shuffle and split
        random.seed(42)
        random.shuffle(all_samples)

        counts = {"train": 0, "valid": 0, "test": 0}

        for i, sample in enumerate(all_samples):
            split = self.get_split(i, len(all_samples))

            # Load and save image
            img = Image.open(sample['img_path'])
            img_array = np.array(img)

            new_img_name = f"{split}_{counts[split]}.png"
            new_img_path = os.path.join(self.target_fold_path, split, new_img_name)

            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img = Image.fromarray(img_array, mode='L').convert('RGB')
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img = Image.fromarray(img_array[:, :, :3], mode='RGB')
            img.save(new_img_path)

            # Store label (multi-line for paragraphs)
            text = sample['label']
            self.charset = self.charset.union(set(text))
            self.gt[split][new_img_name] = text
            counts[split] += 1

        print(f"Paragraph level formatting complete:")
        for split, count in counts.items():
            print(f"  {split}: {count} samples")


def format_ahawp_character(raw_data_path, output_path):
    """Format AHAWP at character level for CTC training"""
    formatter = AHAWPDatasetFormatter(
        level="character",
        raw_data_path=raw_data_path
    )
    formatter.target_fold_path = output_path
    formatter.format()
    return formatter


def format_ahawp_word(raw_data_path, output_path):
    """Format AHAWP at word level for CTC training"""
    formatter = AHAWPDatasetFormatter(
        level="word",
        raw_data_path=raw_data_path
    )
    formatter.target_fold_path = output_path
    formatter.format()
    return formatter


def format_ahawp_paragraph(raw_data_path, output_path):
    """Format AHAWP at paragraph level for seq2seq HAND training"""
    formatter = AHAWPDatasetFormatter(
        level="paragraph",
        raw_data_path=raw_data_path
    )
    formatter.target_fold_path = output_path
    formatter.format()
    return formatter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format AHAWP dataset for HAND framework")
    parser.add_argument("--raw_path", type=str, required=True,
                        help="Path to extracted AHAWP data")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path for formatted output")
    parser.add_argument("--level", type=str, default="character",
                        choices=["character", "word", "paragraph"],
                        help="Dataset level to format")

    args = parser.parse_args()

    formatter = AHAWPDatasetFormatter(
        level=args.level,
        raw_data_path=args.raw_path
    )
    formatter.target_fold_path = args.output_path
    formatter.format()
