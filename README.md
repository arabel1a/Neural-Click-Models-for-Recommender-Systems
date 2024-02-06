# Sim4Rec Response Function
 
# Benchmark
 
## At first run
 
Before you start, you need to download the data available at Misha's Google Drive: [link](https://drive.google.com/drive/folders/17mP753jtLIq4jERKxbQWKjAP994FPT66?usp=drive_link).
 
To work with datasets, you need to create `ContentWise` and `RL4RS` class instances which filter and prepare the data. It takes time, so they can be dumped to pickle for further fast loading. Do something like this:
```
c = ContentWise('/home/arabella/Downloads/data/ContentWiseImpressions/data/ContentWiseImpressions/CW10M-CSV/')
c.dump('cw.pkl')
r = RL4RS('/home/arabella/Downloads/data/rl4rs-dataset/', 'rl4rs_dataset_b_sl.csv')
r.dump('rl4rs.pkl')
```
Warning: It takes ~15 minutes on a 2018's laptop to process and requires ~24GB RAM in total for both datasets.
 
## Usage
To train and evaluate models, load datasets with `ContentWise.load('dump_filename.pkl')` or `RL4RS.load('dump_filename.pkl')`.
 
You will get torch.utils.dataset. Its item is a dict with string keys and numpy array values. See `ReciommendationDataset.__getitem__` for description. See Jupyter Notebooks in benchmark folder to get usage examples.
 
## File structure
```
* benchmark/datasets -- main class with datasets wrapper, and some prepared **datasets.** 
* benchmark/utils -- utility stuff, including models train/evaluate, batch collate function, and so on
* benchmark/evaluated_models -- models with computed metrics goes here to not mess in root directory
* benchmark/*|
|-- Jupyter notebook with different experiments
```

## Wrapping your own dataset
See `RecommendationDataset` class docstring and comments.
Also see comments in ContentWise and RL4RS, it might be helpful.
There is a class DummyData with few users and items. Maybe it can help you understand what's happening.

If you have any further questions or need additional assistance, feel free to ask!

## Models

| notebook name         | experiment name <br>(=result file suffix) | Class name | Paper Name | Comment. |
| --------------------- | ----------------------------------------- | ---------- | ---------- | -------- |
| matrix_factorization  | MatrixFactorization      	| MF                 	| MF | |
| logreg                | LogReg                   	| LogisticRegression 	| Logistic<br>regression | |
| logreg_category_embs  | LogRegCE				 	| LogisticRegression 	| -- | same with `logreg`, but use new embeddings |
| slatewise_attention   | SlatewiseAttention       	| SlatewiseAttention 	| Slate-wise Transformer| |
| attention_plus_gru    | AttentionGRU 			   	| AttentionGRU			| Transformer<br>+ GRU| |
| attention+gru         | AttentionGRU2				| AttentionGRU2			| --- | another version of previous |
| scot                  | SCOT                     	| SCOT					| SCOT | |
| sessionwise_gru       | SessionwiseGRU 		   	| SessionwiseGRU		| Session-wise<br>GRU | |
| slatewise_gru         | SlatewiseGRU             	| SlatewiseGRU       	| Slate-wise<br>GRU | |