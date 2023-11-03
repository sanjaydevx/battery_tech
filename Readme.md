## About
This project is a collection of source code artifacts related to Battery Management Systems (BMS).
Has a port of MATLAB model to Julia. The MATLAB model has used two battery training datasets, one for Li-ion and one for a NMC cell.

The MATLAB model used can be found here - https://www.sae.org/publications/technical-papers/content/2020-01-1181/

## Julia Model

The model, built using Flux.jl, uses the Cell Voltage, Current, Cell Temperature, Capacity and Energy to predict the SOC (state-of-charge) of the cell. The architecture and hyperparameters of the model remained the same. 

The model is a regression model with one output, the SOC. It consists of 3 layers: 5 input neurons, 55 neurons in the hidden layer and 1 output neuron. It was then trained with a batch size of 32 on the exact same dataset with 50 epochs and 3 repetitions. The learning rate was a constant at 0.01 and the optimizer used was ADAM. 

This yielded error percentages (mean error, max error and RMSE) marginally lower than the MATLAB model.

The model can be accessed in the 'src/neural.jl' file. 
The model has been set to train with 3 repetitions of 50 epochs. 
The error stats displayed are with the same configuration. The error stats will be printed out as a dataframe.

To change any hyperparameters, change values while calling the 'define_parameters!' function. 

## Hyperparameter Tuning 

The Julia model was then subjected to hyperparameter tuning to further optimize the accuracy in predictions. 

A decay was introduced in the learning rate and the batch size was varied. The ideal batch size was experimentally found to be 32 and the decay beta1 and beta2 rates found to be 0.9 and 0.95. 

This improved the accuracy of the model with error stats as shown below

![Stats](https://github.com/sanjaydevx/battery_tech/blob/main/images/stats.png)
