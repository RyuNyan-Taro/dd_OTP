from itertools import islice
import json
import os
from pathlib import Path

from loguru import logger
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


BATCH_SIZE = 4
PROGRESS_STEP_DENOM = 100  # Update progress bar every 1 // PROGRESS_STEP_DENOM


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def main():
    # Diagnostics
    logger.info("Torch version: {}", torch.__version__)
    logger.info("CUDA available: {}", torch.cuda.is_available())
    logger.info("CUDA device count: {}", torch.cuda.device_count())

    # Load model
    src_root = Path(__file__).parent.resolve()

    model_path = src_root / "final_model_ipa"

    # load model and processer
    loaded_processor = Wav2Vec2Processor.from_pretrained(model_path)
    loaded_model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Load manifest and process data
    data_dir = Path("data")
    manifest_path = data_dir / "utterance_metadata.jsonl"

    with manifest_path.open("r") as fr:
        items = [json.loads(line) for line in fr]

    # Sort by audio duration for better batching
    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)

    logger.info(f"Processing {len(items)} utterances from {manifest_path}")

    step = max(1, len(items) // PROGRESS_STEP_DENOM)

    # Predict
    predictions = {}
    logger.info("Starting transcription...")
    with torch.no_grad():
        for item in tqdm(items):
            input_values = torch.tensor(item['input_values']).unsqueeze(0)
            # 予測（Logitsを出力）
            logits = loaded_model(input_values).logits

            # 3. デコード (ID配列 -> IPA文字列)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = loaded_processor.batch_decode(predicted_ids)[0].replace('<unk>', ' ')

            predictions[item["utterance_id"]] = transcription

    logger.success("Transcription complete.")

    # Write submission file
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    logger.info(f"Writing submission file to {submission_path}")
    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["phonetic_text"] = predictions[item["utterance_id"]]
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()
