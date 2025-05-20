import argparse
import json
import os
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class AISCode(BaseModel):
    ais_code: str = Field(pattern=r"^\d{6}.\d$")
    description: str = Field()


class AISCodes(BaseModel):
    ais_codes: list[AISCode]


AIS_CODE_JSON_SCHEMA = AISCodes.model_json_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMTPY")
    parser.add_argument("--page-images", required=True)
    parser.add_argument("--file-ext", default=".png")
    parser.add_argument("--extracted-codes", required=True)
    parser.add_argument("--extraction-metadata", required=True)
    args = parser.parse_args()
    return args


def check_page_has_codes(
    *,  # enforce kwargs
    image_url: str,
    client: OpenAI,
) -> Literal["Yes", "No"]:
    output = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Does this document have a header indicating a table with "
                            "columns for AIS 2015, Injury Description, and pFCI? "
                            "Answer with Yes or No."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        extra_body={"guided_choice": ["Yes", "No"]},
        model=client.models.list().data[0].id,
        max_tokens=1024,
        n=1,
        temperature=0,
        seed=42,
    )

    result = output.choices[0].message.content
    return result


def extract_codes(
    *,  # enforce kwargs
    image_url: str,
    client: OpenAI,
) -> list[dict[str, str]]:
    output = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract the AIS codes from the left hand side of this "
                            "document and their corresponding descriptions. "
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        extra_body={"guided_json": AIS_CODE_JSON_SCHEMA},
        model=client.models.list().data[0].id,
        max_tokens=2048,
        n=1,
        temperature=0,
        seed=42,
    )

    result = output.choices[0].message.content
    codes = json.loads(result)["ais_codes"]
    return codes


def main(args):
    client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key,
    )

    pages = []
    for fname in sorted(os.listdir(args.page_images)):
        if not fname.endswith(args.file_ext):
            continue
        pages.append(os.path.join(args.page_images, fname))

    has_codes = []
    for page in tqdm(pages):
        result = check_page_has_codes(
            image_url=f"file://{page}",
            client=client,
        )
        has_codes.append(result)

    # TODO: assumes page names are suffixed with 3-digit page # before file ext
    page_nums = [int(os.path.splitext(os.path.basename(x))[0][-3:]) for x in pages]
    df = pd.DataFrame({"page": page_nums, "page_path": page, "has_codes": has_codes})
    df[["page", "has_codes"]].to_csv(args.extraction_metadata, index=False)

    ais_codes = []
    code_pages = df[df["has_codes"] == "Yes"]
    for page_num, page in zip(tqdm(code_pages["page"]), code_pages["page_path"]):
        codes = extract_codes(
            image_url=f"file://{page}",
            client=client,
        )
        for code in codes:
            code["page"] = page_num
            ais_codes.append(code)

    code_df = pd.DataFrame(ais_codes)
    code_df.to_csv(args.extracted_codes, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
