import argparse
import json
import os
from collections import defaultdict

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class AISCode(BaseModel):
    ais_code: str = Field(pattern=r"^\d{6}.\d$")
    description: str = Field()
    parent_code: str = Field(pattern=r"^\d{6}.\d$")


class AISCodes(BaseModel):
    ais_codes: list[AISCode]


AIS_CODE_JSON_SCHEMA = AISCodes.model_json_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMTPY")
    parser.add_argument("--extracted-codes", required=True)
    parser.add_argument("--page-images", required=True)
    parser.add_argument("--formatted-metadata", required=True)
    parser.add_argument("--formatted-codes", required=True)
    args = parser.parse_args()
    return args


def get_raw_format(
    *,  # enforce kwargs,
    image_url: str,
    page_df: pd.DataFrame,
    client: OpenAI,
) -> str:
    descriptors_str = "\n".join(page_df["description"].to_list())

    output = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Format the following list of injury descriptions into a bulleted list with sub-bullets based on the provided document.\n"
                            f"There may be multiple levels of sub-bullets.\n"
                            f"Use the exact injury descriptions in the provided list and do not provide extra descriptions.\n"
                            # TODO: maybe add something about preserving the order?
                            f"The list of injury descriptions on this page is as follows:\n"
                            f"{descriptors_str}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        model=client.models.list().data[0].id,
        max_tokens=2048,
        n=1,
        temperature=0,
        seed=42,
    )

    raw_format = output.choices[0].message.content
    return raw_format


def get_structured_format(
    *,  # enforce kwargs,
    raw_format: str,
    page_df: pd.DataFrame,
    client: OpenAI,
) -> str:
    page_codes = AISCodes(
        ais_codes=[
            AISCode(
                ais_code=row["ais_code"],
                description=row["description"],
                parent_code="000000.0",
            )
            for _, row in page_df.iterrows()
        ]
    )
    page_json = page_codes.model_dump_json().replace("000000.0", "?")

    output = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f'Using the following formatted list, replace "?" in the given JSON with the parent AIS code if applicable.\n'
                            f"Only consider the immediate parent code when multiple levels of subcodes exist.\n"
                            f'AIS codes are 8 characters, 6 digits, a period ("."), and a final digit.\n'
                            f'If no parent code exists, indicate this with the "000000.0".\n'
                            f"The formatted list is as follows:\n"
                            f"{raw_format}\n\n"
                            f'Replace "?" with the parent AIS code in the following JSON:\n'
                            f"{page_json}"
                        ),
                    },
                ],
            }
        ],
        model=client.models.list().data[0].id,
        max_tokens=2048,
        n=1,
        temperature=0,
        seed=42,
        extra_body={"guided_json": AIS_CODE_JSON_SCHEMA},
    )

    structured_format = output.choices[0].message.content
    return structured_format


def tree_str(
    *,  # enforce kwargs
    node_code: str,
    tree_nodes: dict[str, list[str]],
    code_descriptions: dict[str, str],
    indent: int = 0,
) -> str:
    ret = "  " * indent + "- " + code_descriptions[node_code] + "\n"
    for child_code in tree_nodes[node_code]:
        ret += tree_str(
            node_code=child_code,
            tree_nodes=tree_nodes,
            code_descriptions=code_descriptions,
            indent=indent + 1,
        )
    return ret


def main(args):
    df = pd.read_csv(
        args.extracted_codes,
        dtype={"ais_code": str, "description": str, "page": int},
    )
    client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key,
    )

    metadata = list()
    ais_codes = list()
    for page_num in tqdm(df["page"].drop_duplicates().to_list()):
        image_url = os.path.join(args.page_images, f"page-{page_num:>03}.png")
        image_url = os.path.abspath(image_url)
        image_url = f"file://{image_url}"
        page_df = df.loc[df["page"] == page_num]

        raw_format = get_raw_format(
            image_url=image_url,
            page_df=page_df,
            client=client,
        )

        structured_format = get_structured_format(
            raw_format=raw_format,
            page_df=page_df,
            client=client,
        )

        codes = json.loads(structured_format)["ais_codes"]
        same_len = len(page_df) == len(codes)

        code_descriptions = {x["ais_code"]: x["description"] for x in codes}
        tree_nodes = defaultdict(list)
        for x in codes:
            tree_nodes[x["parent_code"]].append(x["ais_code"])

        reformatted = ""
        for root_code in tree_nodes["000000.0"]:
            reformatted += tree_str(
                node_code=root_code,
                tree_nodes=tree_nodes,
                code_descriptions=code_descriptions,
            )

        metadata.append(
            {
                "page": page_num,
                "raw_format": raw_format,
                "structured_format": structured_format,
                "reformatted": reformatted,
                "same_len": same_len,
            }
        )

        for code in codes:
            code["page"] = page_num
            ais_codes.append(code)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(args.formatted_metadata, index=False)

    code_df = pd.DataFrame(ais_codes)
    code_df.to_csv(args.formatted_codes, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
