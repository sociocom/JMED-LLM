import json
from pathlib import Path

import openai
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from src.utils import (exact_f1_score, get_evaluation_messages,
                       get_first_uppercase_alphabet, get_list_from_string,
                       num_openai_tokens, partial_f1_score, set_seed)


def evaluate(cfg):
    set_seed(cfg.seed)
    if cfg.model_type == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, trust_remote_code=cfg.trust_remote_code)
        if cfg.custom_chat_template:
            tokenizer.chat_template = cfg.custom_chat_template
        if cfg.quant_type == "none":
            model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_model_name_or_path, device_map="auto", trust_remote_code=cfg.trust_remote_code)
        elif cfg.quant_type == "8bit":
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path, 
                device_map="auto", 
                trust_remote_code=cfg.trust_remote_code,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True)
            )
        elif cfg.quant_type == "4bit":
            model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model_name_or_path, 
                device_map="auto", 
                trust_remote_code=cfg.trust_remote_code,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            )
        model.eval()
    elif cfg.model_type == "openai":
        client = openai.OpenAI(api_key=cfg.openai_api_key)

    with torch.inference_mode():
        output_data = {}
        output_data["model_name"] = cfg.pretrained_model_name_or_path
        if cfg.model_type == "huggingface":
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
                if cfg.model_type == "huggingface":
                    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    output_ids = model.generate(input_ids, max_new_tokens=cfg.max_new_tokens, **cfg.generator_kwargs)
                    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                elif cfg.model_type == "openai":
                    input_token_len = num_openai_tokens(messages, model=cfg.pretrained_model_name_or_path)
                    max_tokens = input_token_len + cfg.max_new_tokens
                    response = client.chat.completions.create(
                        model=cfg.pretrained_model_name_or_path,
                        messages=messages,
                        seed=cfg.seed,
                        max_tokens=max_tokens,
                        **cfg.generator_kwargs
                    ).choices[0].message.content
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
            
        output_path = Path(cfg.output_dir).joinpath(f"{cfg.save_file_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
