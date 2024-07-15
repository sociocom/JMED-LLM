# JMED-LLM: Japanese Medical Evaluation Dataset for Large Language Models
日本語医療分野における大規模言語モデルの評価用データセット

This dataset is designed for evaluating large language models in the Japanese medical domain.

## Tasks/Datasets

|Task|Dataset|License|Original Dataset|
|---|---|---|---|
|文章分類|**CRADE**: Case Report Adverse Drug Event|CC-BY-4.0|NTCIR-16 Real-MedNLP (MedTxt-CR)|
||**RRTNM**: Radiology Reports Tumor Nodes Metastasis|CC-BY-4.0|NTCIR17 MedNLP-SC|
||**SNSDS**: Social Network Service Disease Symptom|CC-BY-4.0|NTCIR-13 MedWeb|
|文章ペア分類|**JCSTS**(in progress): Japanese Clinical Semantic Textual Similarity|||
|多肢選択式問題|**JMMLU-Med**: Japanese Massive Multitask Language Understanding in Medical domain|CC-BY-SA-4.0|JMMLU|
|固有表現抽出|**MRPDR**: Medical Report Positive Disease Recognition|CC-BY-4.0|NTCIR-16 Real-MedNLP (MedTxt-CR, MedTxt-RR)|
||**MRMR**: Medical Report Medicine Recognition|CC-BY-4.0|NTCIR-16 Real-MedNLP (MedTxt-CR)|
||**NRNER**(in progress): Nursing Reports Named Entity Recognition|CC-BY-NC-SA-4.0|NursingRecord_NERdataset|
|文章マルチラベリング|**MRICD**(in progress): Medical Record ICD Code|||

### Description
#### Classification
全てのタスクは、100件づつのデータで構成されています。また、均衡なデータセットであるため、Accuracyなどのシンプルな評価指標で適切な評価が可能です。
- **CRADE**: 
症例報告の薬品症状から有害事象（ADE）の可能性を分類
- **RRTNM**: 
読影レポートから癌のTNMステージングを分類
- **SNSDS**: 
模擬Tweetから病気や症状があるかを分類
- **JMMLU-Med**: 
JMMLUに含まれる医療問題のみ
#### Named Entity Recognition
- **MRPDR**: 
症例報告および読影レポートにおいて，患者に実際に認められた症状を抽出
- **MRMR**: 
症例報告および読影レポートにおいて，薬品に関する情報を抽出
- **NRNER**(in progress): 
模擬看護記録からの固有表現抽出


## How to build prompt (example)
### Classification
```python
import pandas as pd


def build_user_prompt(question, options):
    user_prompt = f"質問: {question}\n選択肢:\n"
    alphabet = ["A", "B", "C", "D", "E"]
    for i, option in enumerate(options):
        user_prompt += f"{alphabet[i]}. {option}\n"
    return user_prompt

df = pd.read_csv(dataset_path)
df["options"] = df.filter(regex="option[A-E]").apply(lambda x: x.dropna().tolist(), axis=1)

system_prompt = "与えられた医学に関する質問と選択肢から、最も適切な回答を選択してください。なお、回答には選択肢のアルファベット（例：A）のみを含め、他には何も含めないことを厳守してください。"
for question, options in zip(df["question"], df["options"]):
    user_prompt = build_user_prompt(question, options)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    ... # Generate answer using LLM
```
### Named Entity Recognition
```python
import pandas as pd


df = pd.read_csv(dataset_path)
system_prompt = "与えられた医学に関する質問から、最も適切な回答をしてください。なお、回答にはPythonのリスト形式（例：[\"回答1\", \"回答2\"]）のみを含め、他には何も含めないことを厳守してください。"

for question in df["question"]:
    user_prompt = question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    ... # Generate answer using LLM
```

## License
The license for each dataset follows the terms of the original dataset's license. All other components are licensed under a <a rel="license" href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />