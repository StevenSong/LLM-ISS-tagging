# AIS extraction

We start with a scanned PDF of the AIS 2015 Manual. We aim to extract the AIS codes and their corresponding injury descriptions. The AIS codes are 8 characters, 6 digits, a period, and a final digit. Because the PDF consists of scanned images, we need to use optical character recognition (OCR) to extract text. However, due to the unique formatting of the code tables and various scan-quality issues (e.g. page rotation), we use a vision-language model (VLM) with strong OCR, instruction following, and structured output capabilities to extract and format the AIS codes. The structured output capabilities allow us to directly extract the codes into JSON format.

Our procedure is to first convert the PDF into individual images for each page. We first use the VLM to detect if each page has AIS codes. For pages with codes, we then use the VLM to extract the AIS code with their corresponding injury description. We next have the VLM format the extracted _injury descriptions_ on a given page into a list hierarchy, subcategorizing the descriptions according to the structure presented in the source page. Using the VLM inferred injury hierarchy, we finally have the VLM associate each code with a parent code. We lastly associate metadata to each code according to the AIS coding rules. In total, there are four rounds of VLM inference.

This approach to code extraction and hierarchy formatting was the result of preliminary experiments of different methods. We qualitatively find that our VLM of choice struggled with exactly inferring the structure of codes from the AIS manual. As we prefer the VLM have strong performance in precise text extraction and for simplicity of serving a single VLM, we accept the imprecision in hierarchy formatting. Qualitatively, we find that the inferred hierarchy is reasonable; we evaluate the performance of these extracted and formatted codes in downstream experiments for LLM-based ISS scoring.

For the VLM, we use `Qwen2.5-VL-32B-Instruct` served with `vLLM` on a single A100 80GB GPU. We provide multiple tools for the various steps outlined above. Use the following steps to prepare the codes:

1. Activate virtual environment
1. Convert the PDF to images:
    ```bash
    python convert-pages.py \
    --pdf /path/to/manual.pdf \
    --page-images /path/to/page/images \
    --file-ext .png \
    --dpi 72
    ```
1. Start the VLM inference server in a parallel shell (keep this running while subsequent steps are run):
    ```bash
    vllm serve \
    Qwen/Qwen2.5-VL-32B-Instruct \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt image=1 \
    --allowed-local-media-path /path/to/page/images
    ```
1. Extract AIS codes:
    ```bash
    python extract-codes.py \
    --page-images /path/to/page/images \
    --file-ext .png \
    --extracted-codes /path/to/extracted-codes.csv \
    --extraction-metadata /path/to/extraction-metadata.csv
    ```
1. Format AIS codes:
    ```bash
    python extract-format.py \
    --extracted-codes /path/to/extracted-codes.csv \
    --page-images /path/to/page/images \
    --formatted-codes /path/to/formatted-codes.csv \
    --formatting-metadata /path/to/formatting-metadata.csv
    ```
1. Prepare final metadata:
    ```bash
    python final-metadata.py \
    --input-codes /path/to/formatted-codes.csv \
    --output-codes /path/to/final-codes.csv
    ```