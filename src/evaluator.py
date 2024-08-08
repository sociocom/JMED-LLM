import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (exact_f1_score, get_evaluation_messages,
                       get_first_uppercase_alphabet, get_list_from_string,
                       partial_f1_score, set_seed)


def evaluate(cfg):
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
    if cfg.custom_chat_template:
        tokenizer.chat_template = cfg.custom_chat_template
    model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_model_name_or_path, device_map="auto")
    model.eval()

    with torch.inference_mode():
        output_data = {}
        output_data["model_name"] = cfg.pretrained_model_name_or_path
        output_data["chat_template"] = tokenizer.chat_template
        output_data["generator_kwargs"] = cfg.generator_kwargs
        for task_name in tqdm(cfg.task_names, desc="Processing tasks"):
            dataset_path = Path(cfg.dataset_dir).joinpath(f"{task_name}.csv")
            dataset = pd.read_csv(dataset_path)
            answer_list = dataset["answer"].tolist()
            if task_name in ["jmmlu_med", "crade", "rrtnm", "smdis", "jcsts"]:
                dataset["options"] = dataset.filter(regex="option[A-F]").apply(lambda x: x.dropna().tolist(), axis=1)
            
            task_data = {}
            response_results = []
            for row in tqdm(dataset.itertuples(), desc=f"Processing {task_name} dataset", total=len(dataset)):
                messages = get_evaluation_messages(row, task_name=task_name, use_system_role=cfg.use_system_role)
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                output_ids = model.generate(input_ids, max_new_tokens=cfg.max_new_tokens, **cfg.generator_kwargs)
                response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                response_results.append(response)

            if task_name in ["jmmlu_med", "crade", "rrtnm", "smdis", "jcsts"]:
                predict_results = [get_first_uppercase_alphabet(predict) for predict in response_results]
                if task_name in ["jmmlu_med", "crade", "rrtnm", "smdis", "jcsts"]:
                    score = accuracy_score(answer_list, predict_results)
                    task_data["accuracy"] = score
                if task_name in ["jmmlu_med", "rrtnm", "smdis"]:
                    score = cohen_kappa_score(answer_list, predict_results)
                    task_data["kappa"] = score
                if task_name in ["crade", "jcsts"]:
                    score = cohen_kappa_score(answer_list, predict_results, weights="linear")
                    task_data["kappa"] = score
            elif task_name in ["mrner_disease", "mrner_medicine", "nrner"]:
                answer_list = [get_list_from_string(answer) for answer in answer_list]
                predict_results = [get_list_from_string(predict) for predict in response_results]
                if task_name in ["mrner_disease", "mrner_medicine", "nrner"]:
                    score = exact_f1_score(answer_list, predict_results)
                    task_data["exact_f1"] = score
                if task_name in ["mrner_disease", "mrner_medicine", "nrner"]:
                    score = partial_f1_score(answer_list, predict_results)
                    task_data["partial_f1"] = score

            
            task_data["answer"] = answer_list
            task_data["generated_text"] = response_results
            task_data["predict"] = predict_results
            output_data[task_name] = task_data
            
        output_path = Path(cfg.output_dir).joinpath(f"{cfg.save_file_name}.csv")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
