__all__ = ['pre_train']


import numpy as np
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Union
import os

from .. import calc, models, data

def pre_train():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    model, processor = models.get_model()
    train_dataset, val_dataset = data.prepare_dataset('../datas/train_phon_transcripts.jsonl', processor)

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