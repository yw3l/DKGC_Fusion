import argparse
import json
import torch
from torch.utils.data import DataLoader
from dkgc_model import DKGC
from dkgc_dataloader import DKGCDataset, dkgc_collate_fn
from config import args

def main():
    # 1. Dataloader
    # The entity_id2idx_path is not in the config, so I will hardcode it for now.
    # I will look for it in the data directory.
    entity_id2idx_path = 'data/FB15k237/entity_id2idx.json'
    relation_idx2text_path = 'data/FB15k237/relation_idx2text.json'
    simkgc_entities_path = '../SimKGC/data/FB15k237/FB15k_mid2description.txt'


    with open(entity_id2idx_path, 'r') as f:
        n_entities = len(json.load(f))

    dataset = DKGCDataset(
        triples_path=args.train_path,
        entity_id2idx_path=entity_id2idx_path,
        relation_idx2text_path=relation_idx2text_path,
        simkgc_entities_path=simkgc_entities_path,
        tokenizer_name=args.pretrained_model,
        n_entities=n_entities
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dkgc_collate_fn)

    # 2. Model Initialization
    simkgc_args = argparse.Namespace(pretrained_model=args.pretrained_model, pooling=args.pooling)
    compounde_args = {
        'model': 'CompoundE', 'nentity': n_entities, 'nrelation': 237, 'hidden_dim': 100, 'gamma': 12.0,
        'double_entity_embedding': True, 'double_relation_embedding': False,
        'triple_relation_embedding': True, 'quad_relation_embedding': False
    }
    model = DKGC(simkgc_args, compounde_args)

    # 3. Training
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = torch.nn.MarginRankingLoss(margin=args.additive_margin)
    model.train()

    for epoch in range(args.epochs):
        for pos_batch, neg_batch in dataloader:
            optimizer.zero_grad()
            pos_scores = model(pos_batch)
            neg_scores = model(neg_batch)
            target = torch.ones(pos_scores.size(0))
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(pos_scores, neg_scores, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(pos_scores, neg_scores, target)
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}")

    print("DKGC Fusion Model Training Complete!")


if __name__ == '__main__':
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    main()