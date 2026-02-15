__all__ = ['prepare_dataset']


import librosa
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .. import calc


def prepare_dataset(jsonl_path, processor):

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
