import os
import json
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class SummDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length_src, max_length_tgt, split_type):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length_src = max_length_src
        self.max_length_tgt = max_length_tgt
        self.split_type = split_type
    
    def __len__(self):
        return len(self.dataset)

    def _encode_dialogue(self, dialogue):
        input_ids = [self.tokenizer.bos_token_id]
        for uttr in dialogue:
            uttr_ids = self.tokenizer.encode(uttr, truncation=True, max_length=self.max_length_src, add_special_tokens=False)
            if len(input_ids + uttr_ids) > self.max_length_src - 1:
                break
            input_ids.extend(uttr_ids)
        input_ids.append(self.tokenizer.eos_token_id)
        return torch.tensor(input_ids)

    def __getitem__(self, idx):
        entry = self.dataset[idx]

        if type(entry) == tuple:
            return torch.tensor(entry[0]), torch.tensor(entry[1])

        src = entry['dialogue']
        input_ids = self._encode_dialogue(src)

        if 'summaries' in entry:
            tgts = entry['summaries']
            return input_ids, tgts 

        tgt = entry['summary']
        output_ids = self.tokenizer.encode(tgt, truncation=True, max_length=self.max_length_tgt, return_tensors="pt").squeeze(0)
        
        if 'summary' in entry and self.split_type == "train":
            return input_ids, output_ids

        if 'summary' in entry and self.split_type == "val":
            return input_ids, output_ids, tgt


def collate_fn(batch, pad_token_id=1):
    if not isinstance(batch[0][-1], list):
        if len(batch[0]) == 2:
            input_ids, output_ids = zip(*batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
            output_ids = pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
            return input_ids, output_ids
        else:
            input_ids, output_ids, tgt = zip(*batch)
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
            output_ids = pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
            return input_ids, output_ids, tgt 
    else:
        input_ids, tgts = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, list(tgts)


def load_val_test_dataset(dataset_path, dataset_name, split_type):
    with open(os.path.join(dataset_path, dataset_name, split_type + ".json"), 'r') as rf:
        data = json.load(rf)
    dataset = []
    for sample in data:
        dialogue = [i['role'] + '<eor>' + i['utterance'] + '<eou>' for i in sample['dialogue']]
        if split_type == 'val':
            if 'rouge_avgs' not in sample:
                summary = sample['reference_summ']
            else:
                summary = sample['reference_summ' + str(sample['rouge_avgs'].index(max(sample['rouge_avgs'])) + 1)]
            dataset.append({'dialogue': dialogue, 'summary': summary})
        elif split_type == 'test':
            summaries = [value for key, value in sample.items() if 'reference_summ' in key]
            dataset.append({'dialogue': dialogue, 'summaries': summaries})
        else:
            raise ValueError(f"Invalid split_type: {split_type}")
    return dataset 


def load_train_dataset(mode, dataset_path, dataset_name, few_shot, seed, num_sample):
    if mode == 'pre-training-dap':
        full_path = os.path.join(dataset_path, dataset_name)
        file_paths = [i for i in os.listdir(full_path) if '.pt' in i]
        dataset = []
        for file_path in file_paths:
            dataset.extend(torch.load(os.path.join(full_path, file_path)))
        random.shuffle(dataset)
        return dataset
    elif mode == 'pre-training-top':
        with open(os.path.join(dataset_path, dataset_name), 'r') as rf:
            data = json.load(rf)
        dataset = []
        for sample in data:
            if 'chatgpt_anno_summ' in sample:
                dialogue = [i['added_role'] + '<eor>' + i['utterance'] + '<eou>' for i in sample['dialogue']]
                summary = sample['chatgpt_anno_summ']
                dataset.append({'dialogue': dialogue, 'summary': summary})
            if 'role-rep_named-coref_summ' in sample:
                dialogue = [i['named_coref'] + '<eor>' + i['utterance'] + '<eou>' for i in sample['dialogue']]
                summary = sample['role-rep_named-coref_summ']
                dataset.append({'dialogue': dialogue, 'summary': summary})
            if 'role-rep_cust-serv_summ' in sample:
                dialogue = [i['cust_serv'] + '<eor>' + i['utterance'] + '<eou>' for i in sample['dialogue']]
                summary = sample['role-rep_cust-serv_summ']
                dataset.append({'dialogue': dialogue, 'summary': summary})
        random.shuffle(dataset)
        return dataset
    elif mode == 'fine-tuning':
        with open(os.path.join(dataset_path, dataset_name, 'train.json'), 'r') as rf:
            data = json.load(rf)
        dataset = []
        for sample in data:
            dialogue = [i['role'] + '<eor>' + i['utterance'] + '<eou>' for i in sample['dialogue']]
            if 'rouge_avgs' not in sample:
                summary = sample['reference_summ']
            else:
                summary = sample['reference_summ' + str(sample['rouge_avgs'].index(max(sample['rouge_avgs'])) + 1)]
            dataset.append({'dialogue': dialogue, 'summary': summary})
        if few_shot:
            random.seed(seed)
            dataset = random.sample(dataset, num_sample)
        random.shuffle(dataset)
        return dataset
    else:
        raise ValueError(f"Invalid mode: {mode}")
                

def get_dataloader(args, tokenizer, split_type):
    if split_type == 'train':
        dataset = load_train_dataset(args.mode, args.dataset_path, args.dataset_name, args.few_shot, args.seed, args.num_sample)
    elif split_type == 'val':
        dataset_path = args.val_dataset_path if args.val_dataset_path and args.val_dataset_name else args.dataset_path
        dataset_name = args.val_dataset_name if args.val_dataset_path and args.val_dataset_name else args.dataset_name
        if "-" not in dataset_name:
            dataset = load_val_test_dataset(dataset_path, dataset_name, split_type)
        else:
            dataset_names = dataset_name.split("-")
            dataset = [item for name in dataset_names for item in load_val_test_dataset(dataset_path, name, split_type)]
    elif split_type == 'test':
        dataset = load_val_test_dataset(args.dataset_path, args.dataset_name, split_type)
    else:
        raise ValueError(f"Invalid split_type: {split_type}")

    dataset = SummDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length_src=args.max_length_src,
        max_length_tgt=args.max_length_tgt,
        split_type=split_type
    )

    is_shuffle = split_type == 'train'

    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=is_shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
