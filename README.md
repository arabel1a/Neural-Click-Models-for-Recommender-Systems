# Neural-Click-Models-for-RS
 
# Benchmark
 
## At first run
  
To work with datasets, you need to create `ContentWise` and `RL4RS` class instances which filter and prepare the data. It takes time, so they can be dumped to pickle for further fast loading. Do something like this:
```
c = ContentWise('/home/USER/Downloads/data/ContentWiseImpressions/data/ContentWiseImpressions/CW10M-CSV/')
c.dump('cw.pkl')
r = RL4RS('/home/USER/Downloads/data/rl4rs-dataset/', 'rl4rs_dataset_b_sl.csv')
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

