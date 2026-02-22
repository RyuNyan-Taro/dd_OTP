__all__ = ['pred_saved_model', 'calc_cer_of_predicted']


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import pandas as pd
from tqdm import tqdm

from .. import calc

def pred_saved_model(model_dir, dataset, result_prefix: str):

    # 保存したディレクトリから読み込む
    loaded_processor = Wav2Vec2Processor.from_pretrained(model_dir)
    loaded_model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    print('loaded the model')

    # 1. モデル準備 (CPUで行います)
    loaded_model.to("cpu")
    loaded_model.eval()

    # 2. すべてのサンプルに対して推論を実行
    results = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="推論中"):
            sample = dataset[idx]
            audio_input = sample["input_values"]

            # tensorに変換してバッチ次元を追加 [length] -> [1, length]
            input_values = torch.tensor(audio_input).unsqueeze(0)

            # 予測（Logitsを出力）
            logits = loaded_model(input_values).logits

            # 3. デコード (ID配列 -> IPA文字列)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = loaded_processor.batch_decode(predicted_ids)[0].replace('<unk>', ' ')

            # スコア計算
            score = calc.score_ipa_cer([sample['normalized_text']], [transcription])

            # 結果を保存
            results.append({
                'utterance_id': sample['utterance_id'],
                'predicted': transcription,
                'ground_truth': sample['normalized_text'],
                'cer': score
            })

    # 4. 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    print(f"\n--- 推論完了 ---")
    print(f"総サンプル数: {len(results_df)}")
    print(f"平均CER: {results_df['cer'].mean():.4f}")
    print(f"最小CER: {results_df['cer'].min():.4f}")
    print(f"最大CER: {results_df['cer'].max():.4f}")
    print(f"\n最初の10件の結果:")
    print(results_df.head(10))

    # 5. 結果を保存（オプション）
    results_df.to_csv(f'./{result_prefix}_inference_results.csv', index=False)
    print(f"\n結果を ./{result_prefix}_inference_results.csv に保存しました。")


def calc_cer_of_predicted(model_name: str):

    for _data in ['train', 'val']:
        _file_prefix = {'train': '', 'val': 'valid_'}[_data]
        _result_path = f'./{model_name}_{_file_prefix}results.csv'
        _df = pd.read_csv(_result_path)
        _results = _df.loc[_df.isna().any(axis=1) == False,]

        print(model_name, _data, len(_df), '->', len(_results))
        print(calc.score_ipa_cer(_results['ground_truth'].to_list(), _results['predicted'].to_list()))