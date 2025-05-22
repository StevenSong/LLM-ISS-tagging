import argparse

import pandas as pd

BODY_REGION_MAP = {
    0: "Other Trauma",
    1: "Head",
    2: "Face",
    3: "Neck",
    4: "Thorax",
    5: "Abdomen",
    6: "Spine",
    7: "Upper Extremity",
    8: "Lower Extremity",
    9: "External",
}

STRUCTURE_TYPE_MAP = {
    0: "Not Applicable",
    1: "Whole Area",
    2: "Vessels",
    3: "Nerves",
    4: "Organs (incl. Muscle/Ligament)",
    5: "Skeletal",
    6: "Head - LOC",
    7: "Joints",
    8: "Not Applicable",
    9: "Not Applicable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-codes", required=True)
    parser.add_argument("--output-codes")
    args = parser.parse_args()
    return args


def main(args):
    df = pd.read_csv(
        args.input_codes,
        dtype={
            "ais_code": str,
            "description": str,
            "parent_code": str,
            "page": int,
        },
    )

    body_region_idxs = df["ais_code"].str[:1].astype(int)
    structure_type_idxs = df["ais_code"].str[1:2].astype(int)

    df["body_region"] = body_region_idxs.replace(BODY_REGION_MAP)
    df["structure_type"] = structure_type_idxs.replace(STRUCTURE_TYPE_MAP)
    df["severity"] = df["ais_code"].str[-1].astype(int)

    df.to_csv(args.output_codes, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
