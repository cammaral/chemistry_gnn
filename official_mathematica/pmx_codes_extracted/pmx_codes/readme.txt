This folder contains the code and input files for the machine learning algorithm described in  arXiv:2309.02538v1 (https://doi.org/10.48550/arXiv.2309.02538). Please cite any uses, applications, or modifications of this code as doi: 10.6084/m9.figshare.24082035 and 10.48550/arXiv.2309.02538.

The contents are:
pmx.nb = Mathematica code for training a network to make predictions for unseen molecules.  Uses data files in Mathematica.data. This code should be run handle by handle starting at the top. 

read.trained.net.nb = Mathematica code for reading the weights and biases of a fully trained network and then making predictions (no retraining required).  Uses data files in weight.biases.all25.mols.400.epochs.  These weights and biases were found from running the pmx.nb code with all 25 input training sets for 400 epochs.  This code can be run all at once by selecting all handles and then running it.

See codes and 10.48550/arXiv.2309.02538 for additional details.  