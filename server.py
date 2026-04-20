from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import numpy as np
from evaluations import *
from utils_data.default_tokens import DefaultToken
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers.trainer_pt_utils import nested_numpify
import os
import json


class Server(object):
    def __init__(self, args, eval_loader, log_dir):
        self.args = args
        self.eval_loader = eval_loader
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        self.log_dir = log_dir
        self.tokenizer.model_max_length = self.args.max_length
        
        self.eval_loss_history = []
        special_tokens = dict()
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
        if self.tokenizer.eos_token is None:
            special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
        if self.tokenizer.bos_token is None:
            special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
        if self.tokenizer.unk_token is None:
            special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cpu', output_hidden_states=True, torch_dtype=torch.float16, trust_remote_code=True)
        if args.eval_metrics != 'none':
            self.eval_metrics = args.eval_metrics.split(',')

        if args.filtering_model == 'same':
            self.filtering_model = self.model
            self.filtering_tokenizer = self.tokenizer
        else:
            print(args.filtering_model)
            self.filtering_model = AutoModelForCausalLM.from_pretrained(args.filtering_model, device_map='cpu', output_hidden_states=True, torch_dtype=torch.float16, trust_remote_code=True)
            self.filtering_tokenizer = AutoTokenizer.from_pretrained(args.filtering_model, use_fast=True)
            self.filtering_tokenizer.model_max_length = self.args.max_length
            
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            special_tokens = dict()
            if self.filtering_tokenizer.pad_token is None:
                special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
            if self.filtering_tokenizer.eos_token is None:
                special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
            if self.filtering_tokenizer.bos_token is None:
                special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
            if self.filtering_tokenizer.unk_token is None:
                special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
            self.filtering_tokenizer.add_special_tokens(special_tokens)

        if self.args.peft:
            if args.peft_method == 'lora': 
                from peft import get_peft_model, TaskType, LoraConfig
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
                # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none")
            elif args.peft_method == 'prefix':
                from peft import get_peft_model, TaskType, PrefixTuningConfig
                peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, prefix_projection=False, num_virtual_tokens=10)
            elif args.peft_method == 'p-tuning':
                from peft import get_peft_model, TaskType, PromptEncoderConfig
                peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, encoder_reparameterization_type='MLP', encoder_dropout=0.1, num_virtual_tokens=20)
            elif args.peft_method == 'prompt':
                from peft import get_peft_model, TaskType, PromptTuningConfig
                peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, prompt_tuning_init='RANDOM', num_virtual_tokens=20)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        self.device = torch.device(f'cuda:{self.args.device}')
    
    def prepare_aggregate(self):
        self.model_for_aggregate = deepcopy(self.model)
        for _, v in self.model_for_aggregate.named_parameters():
            if v.requires_grad:
                v.data.zero_()

    
    def online_aggregate(self, client, selected_client_list):
        if self.args.equal_weight:
            weight_array = np.array([1.0 for _ in selected_client_list], dtype=np.float64)
            weight_array /= float(len(selected_client_list))
        else:
            weight_array = np.array([len(client.train_loader) for client in selected_client_list], dtype=np.float64)
            weight_array /= float(np.sum(weight_array))
        
        cur_client_index = 0
        for c in selected_client_list:
            if client.idx == c.idx:
                break
            cur_client_index += 1
        
        cur_weight = weight_array[cur_client_index]
        for k, v in self.model_for_aggregate.named_parameters():
            if v.requires_grad:
                v.data += client.model.state_dict()[k].data * cur_weight
        client.clear_model()

    def finish_aggregate(self):
        self.model = self.model_for_aggregate
    
    def eval(self, cur_round, eval_avg_acc):
        if self.args.eval_metrics == 'none':
            eval_metric = self.eval_loss(cur_round)
            metric_type = "loss"

        elif self.args.eval_metrics in ['acc']:
            eval_metric = self.eval_acc(cur_round)
            metric_type = "acc"

        else:
            eval_metric = self.eval_generate(cur_round)
            metric_type = self.args.generate_eval  # rouge ή bleu

    # ✔ σωστό logging
        eval_avg_acc.append(eval_metric)

    # ===== SAVE CHECKPOINTS =====
        if self.args.save and cur_round > 0:
            save_dir = self.log_dir
            os.makedirs(save_dir, exist_ok=True)

        if len(eval_avg_acc) > 1:
            is_best = (
                (metric_type == "loss" and eval_metric < np.min(eval_avg_acc))
                or
                (metric_type != "loss" and eval_metric > np.max(eval_avg_acc))
            )

            if is_best:
                for file_name in os.listdir(save_dir):
                    if 'best' in file_name:
                        os.remove(os.path.join(save_dir, file_name))

                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, f'model_best_round{cur_round}.bin')
                )

        for file_name in os.listdir(save_dir):
            if 'final' in file_name:
                os.remove(os.path.join(save_dir, file_name))

        torch.save(
            self.model.state_dict(),
            os.path.join(save_dir, f'model_final_round{cur_round}.bin')
        )

    return eval_metric, metric_type
    
    
    def eval_loss(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        loss_total_eval = 0.0
        num_eval = 0
        loss_list = []
        
        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device) 
                }
                outputs = self.model(**batch)
                loss = outputs.loss
                progress_bar_eval.update(1)
                if torch.isnan(loss):
                    loss_list.append('NaN')
                    continue
                loss_list.append(str(loss.item()))
                loss_total_eval += loss
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, loss: {loss_total_eval / num_eval}')
        print()
        print()
        self.eval_loss_history.append(loss_list)
        if self.args.log:
            with open(os.path.join(self.log_dir, 'eval_loss_history.json'), 'w', ) as writer:
                json.dump(self.eval_loss_history, writer)
        self.model = self.model.cpu()
        return (loss_total_eval / num_eval).item()
    
    
    def eval_generate(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        acc_total_eval = 0.0
        num_eval = 0
        
        with torch.inference_mode():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                label_ids = batch['labels'].to(self.device)
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    num_beams=1,
                )
                if self.args.generate_eval == 'rouge':
                    acc_total_eval += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.tokenizer)
                else:
                    acc_total_eval += bleu_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.tokenizer)
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, acc: {acc_total_eval / num_eval}')
        print()
        print()
        self.model = self.model.cpu()
        return acc_total_eval / num_eval


    def eval_acc(self, cur_round):
    self.model = self.model.to(self.device)
    self.model.eval()

    progress_bar_eval = tqdm(range(len(self.eval_loader)))

    all_preds = None
    gt_labels = []

    with torch.inference_mode():
        for batch in self.eval_loader:
            outs = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device)
            )

            gt_labels.append(batch['answer'][0])

            shift_logits = outs.logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].to(self.device).contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view_as(shift_labels)

            loss = loss.mean(dim=1)
            group_loss = loss.split(batch['split_size'])

            preds = torch.stack([torch.argmin(l) for l in group_loss], dim=0)

            preds = nested_numpify(preds).tolist()
            all_preds = preds if all_preds is None else all_preds + preds

            progress_bar_eval.update(1)

    self.model = self.model.cpu()

    return acc_score(all_preds, gt_labels)
