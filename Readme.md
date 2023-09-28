## About
This project is a collection of source code artifacts related to Battery Management Systems (BMS).
Has a port of MATLAB model to Julia. 
The MATLAB model has used two battery training datasets, one for Li-ion and one for a NMC cell.

The MATLAB model used can be found here - https://www.sae.org/publications/technical-papers/content/2020-01-1181/

## Julia Model

The model uses the Cell Voltage, Current, Cell Temperature, Capacity and Energy to predict the SOC (state-of-charge) of the cell. 

The architecture and hyperparameters of the model remained the same. 


The model was trained on the exact same dataset with 50 epochs and 3 repetitions. 
The julia model was then subjected to hyperparameter tuning to further optimize the accuracy in predictions. 
