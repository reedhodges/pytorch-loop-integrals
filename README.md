# Using PyTorch to classify loop integrals by type of divergence

Playing around with training neural networks to classify loop integrals in quantum field theory based on whether they are UV divergent, IR divergent, or not divergent.  Intended as a way for me to practice PyTorch.  This repo will keep evolving as I adapt the models and try new ones.

Check out the Jupyter notebook `pytorch_loop_integrals.ipynb` for the most interesting stuff.  The rest of the files are:

- `engine.py`: all the PyTorch code I wrote to do the work behind the scenes
- `generate_data.py`: generates the training and test data set
- `utils.py`: utility functions for plotting and other miscellaneous things