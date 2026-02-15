__all__ = ["get_model"]


from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC


def get_model(model_to="cpu"):
    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to(model_to)

    return model, processor
