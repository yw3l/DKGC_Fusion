
import torch
from torch.utils.data import Dataset
import json
import random
from transformers import AutoTokenizer

class DKGCDataset(Dataset):
    def __init__(self, triples_path, entity_id2idx_path, relation_idx2text_path, simkgc_entities_path, tokenizer_name, n_entities, max_length=128):
        self.triples = self.load_triples(triples_path)
        self.n_entities = n_entities
        
        with open(entity_id2idx_path, 'r', encoding='utf-8') as f:
            self.entity_id2idx = json.load(f)
        self.idx2entity_id = {v: k for k, v in self.entity_id2idx.items()}

        with open(relation_idx2text_path, 'r', encoding='utf-8') as f:
            self.relation_idx2text = json.load(f)

        with open(simkgc_entities_path, 'r', encoding='utf-8') as f:
            simkgc_entities = json.load(f)
        self.entity_id_to_text = {e['entity_id']: e['entity'] + ' ' + e.get('entity_desc', '') for e in simkgc_entities}

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h_id, r_id, t_id = self.triples[idx]

        # Create a negative sample by corrupting the tail
        neg_t_idx = random.randint(0, self.n_entities - 1)
        neg_t_id = self.idx2entity_id.get(neg_t_idx, "")

        # Prepare positive sample
        pos_sample = self.prepare_sample(h_id, r_id, t_id)
        neg_sample = self.prepare_sample(h_id, r_id, neg_t_id, is_positive=False)

        return pos_sample, neg_sample

    def prepare_sample(self, h_id, r_id, t_id, is_positive=True):
        # CompoundE data (indices)
        h_idx = self.entity_id2idx.get(h_id, -1)
        t_idx = self.entity_id2idx.get(t_id, -1)
        r_idx = int(r_id)

        # SimKGC data (text)
        h_text = self.entity_id_to_text.get(h_id, "")
        r_text = self.relation_idx2text.get(str(r_idx), "")
        t_text = self.entity_id_to_text.get(t_id, "")

        # Tokenize for SimKGC
        hr_tokenized = self.tokenizer(h_text + " " + r_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        t_tokenized = self.tokenizer(t_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        h_tokenized = self.tokenizer(h_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            'compounde_data': torch.LongTensor([h_idx, r_idx, t_idx]),
            'simkgc_hr': {k: v.squeeze(0) for k, v in hr_tokenized.items()},
            'simkgc_t': {k: v.squeeze(0) for k, v in t_tokenized.items()},
            'simkgc_h': {k: v.squeeze(0) for k, v in h_tokenized.items()},
            'is_positive': is_positive
        }

    def load_triples(self, path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((h, r, t))
        return triples

def dkgc_collate_fn(batch):
    pos_batch = [item[0] for item in batch]
    neg_batch = [item[1] for item in batch]
    
    batched_pos = collate_single(pos_batch)
    batched_neg = collate_single(neg_batch)

    return batched_pos, batched_neg

def collate_single(batch):
    batched_data = {}
    batched_data['compounde_data'] = torch.stack([item['compounde_data'] for item in batch])

    simkgc_keys = ['simkgc_hr', 'simkgc_t', 'simkgc_h']
    for key in simkgc_keys:
        batched_data[key] = {
            'input_ids': torch.stack([item[key]['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item[key]['attention_mask'] for item in batch]),
            'token_type_ids': torch.stack([item[key]['token_type_ids'] for item in batch]),
        }
    return batched_data
