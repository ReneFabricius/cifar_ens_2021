# CIFAR ensembling experiments

This project contains several experiments with WeightedLDAEnsemble[^1] performed on CIFAR10 and CIFAR100 datasets.
Experiments in this project process outputs of probabilistic classifiers which constitute the ensemble. We focus on neural network classifiers.
Project[^2] contains scripts for training and saving outputs of the neural networks which we use in these experiments.

Experiments currently contained in this project are the following:
1. Traininng subsets experiment
2. Training subset sizes experiment
3. Validation vs training LDA training experiment
   - Approach one
   - Approach two
4. Zero probability outputs inspection

## Training subsets experiment
Experiment code is in the file *training_subsets_experiment.py*.  
**Experiment on CIFAR10 dataset.**  
This experiment trains WeightedLDAEnsemble on various disjoint subsets of data on which neural networks were trained. 
Size of these subsets is 500, therefore each class has 50 samples and each LDA is trained on 100 samples. 
Three different coupling methods are used: method one and two from [[1]](#1) and Bayes covariant method [[2]](#2).
Goal of this experiment is to determine, whether usage of different training sets for LDA has important effects on the performance of the ensemble.

### Usage
Experiment is performed on a single replication output of the script in[^2].

```
$ python training_subsets_experiment.py -folder replication_folder -train_size 500 -cifar 10 -device cuda
```

Neural network training data are divided into folds of the size at least train_size and ensemble is trained on each of these folds.

### Output
Outputs of the experiment are saved into a folder *exp_subsets_train_outputs*.
This folder contains for each fold:
1. outputs of each pairwise coupling method for the dataset testing set,
2. LDA coefficients of the WeightedLDAEnsemble as a csv,
3. LDA models dump of the WeightedLDAEnsemble.
Apart from these files, the folder also contains summary informations:
1. networks_order - order of the networks in the previous files,
2. net_accuracies.csv - accuracies of the networks conbined in this experiment,
3. accuracies.csv - accuracies of the created ensembles.

## Training subset sizes experiment
Experiment code is in the file *training_subsets_sizes_experiment.py*.  
**Experiment on CIFAR10 dataset.**  
This experiment trains WeightedLDAEnsemble on various subsets of different sizes of data on which neural networks were trained. 
Subsets of the same size are disjoint. Three different coupling methods are used: method one and two from [[1]](#1) and Bayes covariant method [[2]](#2).
Goal of this experiment is to determine, for which size of the LDA training set, the ensemble achieves the best performance.

### Usage
Experiment is performed on a single replication output of the script in[^2].

```
$ python training_subsets_sizes_experiment.py -folder replication_folder -max_fold_rep 30 -device cuda
```
max_fold_rep represents maximum number of folds for each fold size.
Fold sizes start from 8 samples per class and go up to maximally 495 saples per class.
Each next fold is approximately 1.4 times larger, than the previous.

### Output
Outputs oof the experiment are saved into a folder *exp_subsets_sizes_train_outputs*.
This folder contains for each fold size and each fold of this size:
1. outputs of each pairwise coupling method for the dataset testing set,
2. LDA coefficients of the WeightedLDAEnsemble as a csv,
3. LDA models dump of the WeightedLDAEnsemble,
4. indices of the training set samples from which the fold is made.
Apart from these files, the folder also contains summary informations:
1. networks_order - order of the networks in the previous files,
2. net_accuracies.csv - accuracies of the networks conbined in this experiment,
3. accuracies.csv - accuracies of the created ensembles.

## Validation vs training LDA training experiment
This experiment focuses on the question, whether training LDA on the same set of data as the neural networks were trained on 
has any adverse effects to the performance of the ensemble as opposed to training on a different set, not presented to the networks during the training.  
Experimment was performed with two slightly different approaches.

### Aproach one
Experiment code is in the file *base_ensembling_experiment.py*.  
**Experiment on CIFAR10 dataset.**  
This experiment was performed in 30 replications. In each replication a set of 500 samples from CIFAR10 training set was randomly chosen, 
with each class represented equally, this set is reffered to as validation set. This set was extracted from the CIFAR10 training set. 
Three neural networks were trained on the reduced training set. These networks were then combined using WeightedLDAEnsemble.
For each replication, the ensemble was built twice. First LDA was trained on the extracted validation set and second on a randomly chosen set of 500 samples 
from the neural networks training set. These LDA training sets are referred to as vt and tt respectively.

#### Usage
Experiment is performed on the result of several replications of the script in[^2].
```
$ python base_ensembling_experiment.py -folder experiment_root_folder -repl 30 -cifar 10 -device cuda
```

experiment_root_folder is the folder, which contains folders corresponding to individual replications.

#### Output
Experiment produces output in each processed replication folder. This output is placed inside *comb_outputs* folder, which contains following:
1. *networks_order* file, which lists order of the processed networks in all the stored tensors,
2. *train_training* folder,
3. *val_training* folder.
*train_training* folder corresponds to ensemble trained on a random subset of the networks training set,
*val_training* folder corresponds to ensemble trained on validation set. 
Both these folders contain stored WeightedLDAEnsemble model and outputs of all the tested coupling methods.
*train_training* folder also contains indices into the neural networks training set, which constitute LDA training set.
Apart from these replication-specific outputs, the experiment also produces summary outputs *ensemble_accuracies.csv* and *net_accuracies.csv*.
Both of these files are stored in the experiment root folder.

### Aproach two
Experiment code is in the file *half_train_ensembling_experiment.py*.  
**Experiment on both CIFAR10 and CIFAR100 datasets.**  
This experiment differs from the previous one in the neural networks training part. 
In this case, the networks were trained on half of the original training set. The remainder of the training set was extracted as a validation set.
This enabled us to train several ensembles on both the training set and the validation set in each replication.
Experiment on CIFAR10 was performed in 1 replication and experimennt on CIFAR100 in 10 replications.
This is due to 10 times more classes in CIFAR100 and thus a need for 10 times larger LDA training set in order to maintain constant 50 samples for class in LDA models training.

#### Output
Output is simillar to that of the Aproach one. Difference is only in that both *train_training* and *val_training* folders contain outputs of several folds as well as indices
of samples making up these folds.

## Zero probability outputs inspection
This quick debugging script came up as a result of observing a strange behavior of negative log likelihood (nll) score when evaluating outputs of the ensembles.
WeightedLDAEnsemble is able to produce zero probabilities for some classes. In computing the nll score, we therefore had to set a minimum probability threshold and replace
zero probability outputs by this value. It came to our attention, that value of this minimum probability threshold has a great impact on the resulting nll score.
We therefore decided to examine whether these zero probabilities for corect class in the ensemble outputs are product of a bug or just a specific behavior of the ensemble.
So far we didn't find any bug that would explain this behavior. We will look into a posibile way of avoiding these zero proobability outputs in the future.


[^1]: https://github.com/ReneFabricius/weighted_ensembles
[^2]: https://github.com/ReneFabricius/cifar_train_2021

## References
<a id="1">[1]</a> 
Wu, Ting-Fan, Chih-Jen Lin, and Ruby C. Weng. 
"Probability estimates for multi-class classification by pairwise coupling."
Journal of Machine Learning Research 5.Aug (2004): 975-1005.
<a id="2">[2]</a>
Šuch, Ondrej, and Santiago Barreda. 
"Bayes covariant multi-class classification." 
Pattern Recognition Letters 84 (2016): 99-106.
