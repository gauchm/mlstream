# mlstream

Machine learning for streamflow prediction.

Oxford English Dictionary:

>  *maelstrom*, n. &nbsp;  /ˈmeɪlˌstrɑm/
>
> 1. A powerful whirlpool, originally (usually Maelstrom) one in the Arctic Ocean off the west coast of Norway, which was formerly supposed to suck in and destroy all vessels within a wide radius.
> 2. Any state of turbulence or confusion; a swirling mass of small objects.

## Usage

This project is still work in progress.
The idea is to create an easy way of training machine learning streamflow models:
Just provide your data, select a model (or provide your own), and get the predictions.

### Training
```python
exp = Experiment(data_path, is_train=True, run_dir=run_dir,
                 start_date='01012000', end_date='31122015',
                 basins=train_basin_ids, 
                 forcing_attributes=['precip', 'tmax', 'tmin'],
                 static_attributes=['area', 'regulation'])

exp.set_model(model)
exp.train()
```

### Inference
```python
run_dir = Path('./experiments')
exp = Experiment(data_path, is_train=False, 
                 run_dir=run_dir, 
                 basins=test_basin_ids,
                 start_date='01012016', end_date='31122018')
model.load(run_dir / 'model.pkl')
exp.set_model(model)  
results = exp.predict()
```

