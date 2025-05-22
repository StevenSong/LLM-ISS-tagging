import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ais-codes", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()
    return args


def prepare_full_descriptions(df: pd.DataFrame) -> list[str]:
    descriptions = []
    for code, row in df.iterrows():
        curr = row

        description = curr["description"]
        while curr["parent_code"] != "000000.0":
            curr = df.loc[curr["parent_code"]]
            description = curr["description"] + ". " + description

        if row["body_region"] == "Other Trauma":
            prefix = ""
        elif row["structure_type"] != "Not Applicable":
            prefix = f"{row['body_region']} Injury ({row['structure_type']}): "
        else:
            prefix = f"{row['body_region']} Injury: "

        description = prefix + description

        descriptions.append(description)

    return descriptions


def main(args):
    assert not os.path.exists(args.output_h5)

    df = pd.read_csv(args.ais_codes, dtype={"ais_code": str, "parent_code": str})
    assert not df["ais_code"].duplicated().any()
    df = df.set_index("ais_code")

    passage_prefix = ""
    passages = prepare_full_descriptions(df)
    codes = df.index.to_list()

    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
    model = model.eval().to("cuda")

    with torch.inference_mode(), h5py.File(args.output_h5, "w") as h5:
        for i in trange(0, len(passages), args.batch_size):
            batch_passages = passages[i : i + args.batch_size]
            batch_codes = codes[i : i + args.batch_size]
            batch_embs = model.encode(
                batch_passages,
                instruction=passage_prefix,
                max_length=args.max_length,
            )
            batch_embs = batch_embs.to("cpu")

            for code, emb in zip(batch_codes, batch_embs):
                h5[code] = emb


if __name__ == "__main__":
    args = parse_args()
    main(args)
