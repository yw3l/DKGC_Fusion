import argparse

import torch

from torch.utils.data import DataLoader

from dkgc_model import DKGC

from dkgc_dataloader import DKGCDataset, dkgc_collate_fn


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # Add arguments for file paths

    parser.add_argument('--triples_path', type=str, default='data/FB15k237/train.txt')

    parser.add_argument('--entity_id2idx_path', type=str, default='data/FB15k237/entity_id2idx.json')

    parser.add_argument('--relation_idx2text_path', type=str, default='data/FB15k237/relation_idx2text.json')

    parser.add_argument('--simkgc_entities_path', type=str, default='../SimKGC/data/FB15k237/FB15k_mid2description.txt')

    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')

    # Training hparams

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--margin', type=float, default=1.0)

    return parser.parse_args(args)


def main(args):
    # 1. Dataloader

    with open(args.entity_id2idx_path, 'r') as f:

        n_entities = len(json.load(f))

    dataset = DKGCDataset(

        triples_path=args.triples_path,

        entity_id2idx_path=args.entity_id2idx_path,

        relation_idx2text_path=args.relation_idx2text_path,

        simkgc_entities_path=args.simkgc_entities_path,

        tokenizer_name=args.tokenizer_name,

        n_entities=n_entities

    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dkgc_collate_fn)

    # 2. Model Initialization

    simkgc_args = argparse.Namespace(pretrained_model='bert-base-uncased', pooling='cls')

    compounde_args = {

        'model': 'CompoundE', 'nentity': n_entities, 'nrelation': 237, 'hidden_dim': 100, 'gamma': 12.0,

        'double_entity_embedding': True, 'double_relation_embedding': False,

        'triple_relation_embedding': True, 'quad_relation_embedding': False

    }

    model = DKGC(simkgc_args, compounde_args)

    # 3. Training

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)

    model.train()

    for epoch in range(args.epochs):

        for pos_batch, neg_batch in dataloader:
            optimizer.zero_grad()

            pos_scores = model(pos_batch)

            neg_scores = model(neg_batch)

            target = torch.ones(pos_scores.size(0))

            loss = criterion(pos_scores, neg_scores, target)

            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}")

    print("DKGC Fusion Model Training Complete!")


if __name__ == '__main__':
    main(parse_args())
