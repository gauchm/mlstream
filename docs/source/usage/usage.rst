Usage
=====

This project is still work in progress. The idea is to create an easy way of training machine learning streamflow models: Just provide your data, select a model (or provide your own), and get the predictions.

Training
--------
::

  exp = Experiment(data_path, is_train=True, run_dir=run_dir,
                   start_date='01012000', end_date='31122015',
                   basins=train_basins, 
                   forcing_attributes=['precip', 'tmax', 'tmin'],
                   static_attributes=['area', 'regulation'])
  exp.set_model(model)
  exp.train()

Inference
---------
::

  run_dir = Path('./experiments')
  exp = Experiment(data_path, is_train=False, 
                   run_dir=run_dir, basins=['01ABC123', '02DEF123'],
                   start_date='01012016', end_date='31122018')
  model.load(run_dir / 'model.pkl')
  exp.set_model(model)  
  results = exp.predict()

