# JMED-LLM: Japanese Medical Evaluation Dataset for Large Language Models
**JMED-LLM** (**J**apanese **M**edical **E**valuation **D**ataset for **L**arge **L**anguage **M**odels) は，日本語の医療分野における大規模言語モデルの評価用データセットです．JMED-LLMは，奈良先端科学技術大学院大学ソーシャル・コンピューティング研究室がシェアードタスクの開催などを通じて構築してきたデータセットを中心に，日本語の医療言語処理タスク向けに公開されている既存のオープンなデータセットをLLM評価に適したタスクに変換し統合したデータセットです．生成AIの医療応用のための性能評価を目的としており，医療応用のタスクごとに適したLLMを選択可能とすることを目指し今後も継続的にデータセットの拡充を行っていきます．

## 関連資料
- [JMED-LLM: 日本語医療LLM評価データセットの公開(2024/07/25)](https://speakerdeck.com/fta98/jmed-llm-ri-ben-yu-yi-liao-llmping-jia-detasetutonogong-kai) (スライド記載の内容は，データセット内容やLLMの性能など一部古い情報が含まれます)


## Tasks/Datasets

|Task|Dataset|License|Original Resouce|
|---|---|---|---|
|質問応答|**JMMLU-Med**|CC-BY-SA-4.0|JMMLU|
|固有表現抽出|**MRNER-disease**|CC-BY-4.0|NTCIR-16 Real-MedNLP MedTxt-CR Corpus, MedTxt-RR Corpus|
||**MRNER-medicine**|CC-BY-4.0|NTCIR-16 Real-MedNLP MedTxt-CR Corpus|
||**NRNER**|CC-BY-NC-SA-4.0|NursingRecord_NERdataset|
|文書分類|**CRADE**|CC-BY-4.0|NTCIR-16 Real-MedNLP MedTxt-CR Corpus|
||**RRTNM**|CC-BY-4.0|NTCIR-17 MedNLP-SC MexTxt-RR Corpus|
||**SMDIS**|CC-BY-4.0|NTCIR-13 MedWeb Corpus|
|文類似度|**JCSTS**|CC-BY-NC-SA-4.0|Japanese-Clinical-STS|

## Description
### 質問応答
- **JMMLU-Med (Japanese Massive Multitask Language Understanding in Medical domain):** JMMLUから医療分野の問題のみを抽出し集約した，質問応答タスクである．professional\_medicine，medical\_genetics，clinical\_knowledge，anatomy，college\_medicine の五つの専門分野を対象とし，20問ずつで構成されている．

### 固有表現抽出
- **MRNER-disease (Medical Report Named Entity Recognition for positive disease):** 症例報告および読影レポートから患者に実際に認められた症状を抽出するタスクである．病変・症状エンティティのうち，certainty属性がpositiveのものを抽出対象とする．症例報告と読影レポート50件ずつで構成されている．

- **MRNER-medicine (Medical Report Named Entity Recognition for medicine):** 症例報告から医薬品に関する情報を抽出するタスクである．
MRNER-diseaseと同様のデータセットを用いているが，読影レポートには対象のエンティティが含まれていないため利用しない．

- **NRNER (Nursing Record Named Entity Recognition):** 模擬看護記録から患者に実際に認められた症状および薬品に関する情報を抽出するタスクである．
MRNERと同様のタスクだが，データセットのライセンスが異なるため別タスクとして設計した．

### 文書分類
- **CRADE (Case Report Adverse Drug Event):** 症例報告における薬品および症状から有害事象 (ADE) の可能性を分類するタスクである．Diseaseタグ（病名・症状を示す）のデータ48件とMedicineタグ（医薬品に関する情報を示す）のデータ52件から構成されている．

- **RRTNM (Radiology Report Tumor Nodes Metastasis):** 肺がん患者の読影レポートから，がんのTNM分類を予測するタスクである．Tタグ15件，Nタグ40件，Mタグ45件から構成されている．

- **SMDIS (Social Media Disease):** 模擬Tweetから投稿者または周囲の人々の病気や症状の有無を分類するタスクである．influenza，diarrhea，hayfever，coughタグがそれぞれ13件，headache，fever，runnynose，cold タグが12件ずつで構成されている．

### 文類似度
- **JCSTS (Japanese Clinical Semantic Textual Similarity):** 2文の意味的類似度を判定するタスク（STS）の医療版であり，症例報告を扱う．

## Leaderboard
多肢選択式タスクは，括弧外がkappa係数（CRADEとJCSTSは線形重み付き），括弧内がaccuracy．

固有表現抽出タスクは，括弧外が部分一致F1，括弧内が完全一致F1．
|Model|JMMLU-Med|MRNER-disease|MRNER-medicine|NRNER|CRADE|RRTNM|SMDIS|JCSTS|
|---|---|---|---|---|---|---|---|---|
|gpt-4o-2024-08-06|0.82(0.87)|0.54(0.15)|0.42(0.26)|0.39(0.20)|0.54(0.53)|0.85(0.90)|0.76(0.88)|0.60(0.48)|
|gpt-4o-mini-2024-07-18|0.77(0.83)|0.48(0.13)|0.52(0.32)|0.48(0.25)|0.21(0.37)|0.58(0.71)|0.56(0.78)|0.57(0.51)|

## How to evaluate
1. レポジトリのクローン
    ```
    git clone https://github.com/sociocom/JMED-LLM.git
    ```
2. 必要なパッケージのインストール
    ```
    poetry install
    ```
3. config_template.yamlをコピーし設定ファイルを作成（評価対象モデルやプロンプトなど実験設定を必要に応じて変更してください）
    ```
    cp configs/config_template.yaml configs/your_config.yaml
    ```
4. 評価スクリプトの実行
    ```
    poetry run python scripts/evaluate.py --cfg configs/your_config.yaml
    ```

## License
The license for each dataset follows the terms of the original dataset's license. All other components are licensed under a <a rel="license" href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />
