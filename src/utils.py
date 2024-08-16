import random

import numpy as np
import tiktoken
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_evaluation_messages(row, task_name, use_system_role=True):
    question = row.question

    if task_name in ["jmmlu_med", "crade", "rrtnm", "smdis", "jcsts"]:
        options = row.options
        system_prompt = "与えられた医学に関する質問と選択肢から、最も適切な回答を選択してください。なお、回答には選択肢のアルファベット（例：A）のみを含め、他には何も含めないことを厳守してください。"
        user_prompt = f"質問: {question}\n選択肢:\n"
        alphabet = ["A", "B", "C", "D", "E", "F"]
        for i, option in enumerate(options):
            user_prompt += f"{alphabet[i]}. {option}\n"
    elif task_name in ["mrner_disease", "mrner_medicine", "nrner"]:
        user_prompt = question
        system_prompt = "与えられた医学に関する質問から、最も適切な回答をしてください。なお、回答にはPythonのリスト形式（例：['回答1', '回答2']）のみを含め、他には何も含めないことを厳守してください。"
    else:
        raise ValueError(f"task_nameが存在しません: {task_name}")
    
    if use_system_role:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
        ]
    return messages

def get_first_uppercase_alphabet(text):
    uppercase_alphabet = "ABCDEF"
    for char in text:
        if char in uppercase_alphabet:
            return char
    return "None"

def get_list_from_string(text):
    try: 
        eval_text = eval(text)
        if isinstance(eval_text, list):
            return eval_text
        else:
            return []
    except: 
        return []

def exact_f1_score(answer_list, predict_list):
    f1_score = 0
    for answer, predict in zip(answer_list, predict_list):
        answer = set(answer)
        predict = set(predict)
        exact_count = answer & predict

        recall = len(exact_count) / len(answer)
        precision = len(exact_count) / len(predict) if len(predict) != 0 else 0
        f1_score += (2 * recall * precision) / (recall + precision) if (recall + precision) != 0 else 0
    return f1_score / len(answer_list)

def partial_f1_score(answer_list, predict_list):
    f1_score = 0
    for answer, predict in zip(answer_list, predict_list):
        answer = set(answer)
        predict = set(predict)
        partial_count = 0
        for answer_v in answer:
            for predict_v in predict:
                if (answer_v in predict) or (predict_v in answer):
                    partial_count += 1
                    break

        recall = partial_count / len(answer)
        precision = partial_count / len(predict) if len(predict) != 0 else 0
        f1_score += (2 * recall * precision) / (recall + precision) if (recall + precision) != 0 else 0
    return f1_score / len(answer_list)

def num_openai_tokens(messages, model):
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens
