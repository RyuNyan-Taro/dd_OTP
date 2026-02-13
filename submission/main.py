from itertools import islice
import json
import os
from pathlib import Path

import librosa
from loguru import logger
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

BATCH_SIZE = 16  # Increase for GPU batching
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

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    if torch.cuda.is_available():
        logger.info("CUDA device name: {}", torch.cuda.get_device_name(0))
        logger.info("CUDA device capability: {}", torch.cuda.get_device_capability(0))
        # Clear GPU cache
        torch.cuda.empty_cache()

    # Load model
    src_root = Path(__file__).parent.resolve()
    model_path = src_root / "final_model_ipa"

    # Load model and processor
    logger.info("Loading model and processor from {}", model_path)
    loaded_processor = Wav2Vec2Processor.from_pretrained(model_path)
    loaded_model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Move model to device and set to eval mode
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    logger.info("Model loaded on device: {}", device)

    # Load manifest and process data
    data_dir = Path("data")
    manifest_path = data_dir / "utterance_metadata.jsonl"

    with manifest_path.open("r") as fr:
        items = [json.loads(line) for line in fr]

    # Sort by audio duration for better batching (decreasing order for efficiency)
    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)
    logger.info(f"Processing {len(items)} utterances from {manifest_path}")

    # Predict with batching
    predictions = {}
    logger.info("Starting transcription with batch size: {}", BATCH_SIZE)

    with torch.no_grad():
        for batch_items in tqdm(batched(items, BATCH_SIZE), total=(len(items) + BATCH_SIZE - 1) // BATCH_SIZE,
                                desc="Transcribing"):
            try:
                # Load audio for all items in batch
                batch_audio = []
                batch_utterance_ids = []

                for item in batch_items:
                    try:
                        audio, _ = librosa.load(data_dir / item["audio_path"], sr=16000)
                        batch_audio.append(audio)
                        batch_utterance_ids.append(item["utterance_id"])
                    except Exception as e:
                        logger.error(f"Error loading audio for {item['utterance_id']}: {e}")
                        predictions[item["utterance_id"]] = ""

                if not batch_audio:
                    continue

                # Process batch through model
                inputs = loaded_processor(batch_audio, sampling_rate=16000, return_tensors="pt", padding=True)

                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Run inference
                logits = loaded_model(**inputs).logits

                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                transcriptions = loaded_processor.batch_decode(predicted_ids)

                # Store results
                for utterance_id, transcription in zip(batch_utterance_ids, transcriptions):
                    predictions[utterance_id] = transcription.replace('<unk>', ' ').strip()

                # Clear GPU cache after each batch to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                for item in batch_items:
                    predictions[item["utterance_id"]] = ""

    logger.success("Transcription complete.")

    # Write submission file
    submission_format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    logger.info(f"Writing submission file to {submission_path}")

    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with submission_format_path.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            item["phonetic_text"] = predictions.get(item["utterance_id"], "")
            fw.write(json.dumps(item) + "\n")

    logger.success("Done.")


if __name__ == "__main__":
    main()