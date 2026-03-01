__all__ = ["get_model"]


from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC


def get_model(model_to="cpu"):
    model_id = "facebook/wav2vec2-xls-r-300m"
    vocab_path = 'mod_vocab.json'

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_id,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    model.to(model_to)

    return model, processor
