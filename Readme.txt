I used this repo for my masters thesis on domain adaptation in industrial contexts. For running
the scripts you will need to download the following datasets:

CWRU: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
UCSD: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
Office+Caltech: https://mega.nz/folder/AaJTGIzD#XHM2XMsSd9V-ljVi0EtvFg

Relevant scripts are:

1. code/supervised_only/CWRU_extended_task.py : Run in order to to train and evaluate a model to classify the 10 category CWRU task

2. code/eval/eval_classicaDA.py : For evaluating domain adaptation methods on office-caltech dataset with 1NN classifier

3. code/CDAN/CWRU_extended_task.py to reproduce CDAN results on CWRU

4. code/Deep_MMD/CWRU_extended_task.py to reproduce results from Deep MMD on CWRU

5. code/Domain_critic/CWRU_extended_task.py to reproduce results from DANN on CWRU

6. code/manual_feature_extraction/CWRU_extended_task.py to reproduce results of classical methods on ts_fresh features

7. code/manual_feature_extraction/chemical.py to reproduce results of classical methods on UCSD in the normal problem
    setup

8. code/manual_feature_extraction/chemical.py to reproduce results of classical methods on UCSD in the sequential problem
    setup

The repo also includes an implementation of generalised singular value decomposition using only
scipy as a dependency.