__all__ = ['pre_train']


import numpy as np
import os
from transformers import TrainingArguments, Trainer

from .. import calc, models, data

def pre_train():
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    model, processor = models.get_model()
    train_dataset, val_dataset = data.prepare_dataset('../datas/train_phon_transcripts.jsonl', processor)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    data_collator = data.DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = _create_trainer(model, processor, train_dataset, val_dataset, data_collator)

    trainer.train()

    output_dir = "./final_model_ipa"

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print(f"モデルとプロセッサを {output_dir} に保存しました。")


def _create_trainer(model, processor, train_dataset, val_dataset, data_collator):

    def _compute_metrics(pred):
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

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    training_args = TrainingArguments(
        output_dir="./wav2vec2-ipa-checkpoints",
        disable_tqdm=False,
        group_by_length=True,
        remove_unused_columns=False,
        use_mps_device=False,
        use_cpu=True,
        per_device_train_batch_size=4,
        num_train_epochs=30,
        eval_strategy='steps',
        fp16=False,
        logging_first_step=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=_compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.feature_extractor,
    )

    return trainer
