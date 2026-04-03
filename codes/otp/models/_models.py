__all__ = ["get_model"]


from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    Wav2Vec2Config


def get_model(model_to="cpu"):
    # model_id = "facebook/wav2vec2-xls-r-300m"
    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    # Configの読み込みと調整
    config = Wav2Vec2Config.from_pretrained(model_id)
    config.update({
        "mask_time_prob": 0.1,  # 時間軸方向にマスクする確率（少し上げる）
        "mask_time_length": 10,  # マスクする長さ
        "mask_feature_prob": 0.05,  # 周波数軸方向にマスクする確率
        # "ctc_loss_reduction": "mean",
        # "ctc_zero_infinity": True,  # ロスの計算を安定させる
        # "pad_token_id": 0,  # Tokenizerに合わせて設定
        # "vocab_size": 42  # 実際のvocabサイズに合わせる
    })

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(model_id, config=config)
    model.to(model_to)

    return model, processor
