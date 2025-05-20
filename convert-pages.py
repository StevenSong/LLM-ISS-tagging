import argparse
import os

import pymupdf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--page-images", required=True)
    parser.add_argument("--file-ext", default=".png")
    parser.add_argument("--dpi", type=int, default=72)
    args = parser.parse_args()
    return args


def main(args):
    doc = pymupdf.open(args.pdf)
    for page in tqdm(doc):
        im = page.get_pixmap(dpi=args.dpi)
        page_name = f"page-{page.number:>03}.{args.file_ext}"
        im.save(os.path.join(args.page_images, page_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
