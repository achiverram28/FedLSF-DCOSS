## FedLSF -  Federated Local Graph Learning via Specformers

## Instructions
Run `python main.py`

## Datasets
- Cora => `--dataset="cora"`
- Citeseer => `--dataset="citeseer"`
- Amazon Photo => `--dataset="photo"`
- Chameleon => `--dataset="chameleon"`
- Squirrel => `--dataset="squirrel"`

## Non-IID concentration α
`--alpha=<value>`

## Dropouts
`--tran_dropout=<value>` , `--feat_dropout=<value>`, `--prop_dropout=<value>`

## Normalisations
- LayerNorm => `--norm="layer"`

## Number of Clients
`--n_trainer=<value>`

