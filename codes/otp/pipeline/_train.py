__all__ = ['pre_train']


import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Union

from .. import calc

def pre_train():
    import os
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    # 1. 音の特徴を抽出する部分を読み込む
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    # 2. 文字（IPA）を扱う部分を読み込む
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)

    # 3. 両者を合体させて Processor を作る（ここが解決の鍵！）
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 4. モデル本体を読み込む
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to("cpu")

    import librosa

    def prepare_dataset(jsonl_path):

        def _tokenize_labels(batch):
            # processor を使ってテキストを ID 配列に変換
            # ※ normalized_text をターゲット（labels）としてエンコードします
            with processor.as_target_processor():
                batch["labels"] = processor(batch["normalized_text"]).input_ids
            return batch

        # 1. JSONLを読み込む
        df = pd.read_json(jsonl_path, lines=True)
        df['normalized_text'] = df['phonetic_text'].apply(calc.normalize_ipa)

        df['audio_path'] = df['audio_path'].apply(lambda x: '../datas/' + '/' + x)
        df = df.rename(columns={'audio_path': 'audio'})

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

        # 2. 手動デコード用の関数
        def _manual_map(example):
            try:
                # audio_path カラムにあるパスから直接読み込む
                # sr=16000 を指定してリサンプリングも同時に行う
                speech_array, _ = librosa.load(example["audio"], sr=16000)
                example["input_values"] = speech_array
                return example
            except Exception as e:
                # 読み込めないファイルがあった場合は None を入れて後で filter する
                example["input_values"] = None
                return example

        # 3. Dataset作成（cast_column はしない！）
        train_ds = Dataset.from_pandas(train_df).map(_manual_map).map(_tokenize_labels)
        val_ds = Dataset.from_pandas(val_df).map(_manual_map).map(_tokenize_labels)

        return train_ds, val_ds

    train_dataset, val_dataset = prepare_dataset('../datas/train_phon_transcripts.jsonl')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # すでに準備してある "input_values" を直接取り出す
            input_features = [{"input_values": feature["input_values"]} for feature in features]

            batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

            # ラベル側の処理 (labels はそのままでOKなはずです)
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.pad(
                labels=label_features,  # ここを labels_batch から labels に変更
                padding=self.padding,
                return_tensors="pt"
            )

            # -100 でパディングを置き換えて損失計算時に無視するようにする
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # パディング（-100）を無視してデコード
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # 評価スクリプトはリスト形式で受け取るため、そのまま渡す
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # 提供されたスクリプトの score_ipa_cer を利用
        cer = calc.score_ipa_cer(label_str, pred_str)

        return {"cer": cer}

    from transformers import TrainingArguments, Trainer
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    training_args = TrainingArguments(
        output_dir="./wav2vec2-ipa-checkpoints",
        disable_tqdm=False,
        group_by_length=True,
        remove_unused_columns=False,
        use_mps_device=False,  # CPUを使用
        use_cpu=True,
        per_device_train_batch_size=8,
        num_train_epochs=30,
        eval_strategy='steps',
        fp16=False,
        logging_first_step=True,
        save_steps=500,
        eval_steps=500,  # 500ステップごとに検証データでCERを計算
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        metric_for_best_model="cer",  # CERが最も低いモデルを保存
        greater_is_better=False,
        load_best_model_at_end=True,  # 学習終了時にベストモデルを読み込む
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,  # ここで自作メトリクスを登録
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    # 保存先のパス
    output_dir = "./final_model_ipa"

    # モデル本体を保存
    trainer.save_model(output_dir)

    # プロセッサ（FeatureExtractor + Tokenizer）も忘れずに保存
    # これがないと、推論時に「どのIDがどのIPAか」がわからなくなります
    processor.save_pretrained(output_dir)

    print(f"モデルとプロセッサを {output_dir} に保存しました。")