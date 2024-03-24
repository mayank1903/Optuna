# Optuna
This repo contains all the experiments related to Optuna library in python

Intially to get the environment, run the below code 

```
conda env create -f environment.yml
conda activate optuna
```

__Experiments folder consists of all the .ipynb notebooks related to Optuna and its understanding__

1. <ins>Hyper parameter tuning using optuna</ins> --> hyperparameter_tuning_using_optuna.py
Optuna is widely used for hyper-parameter tuning which uses various samplers for tuning the parameters

The above script contains the hyper-parameter tuning on MNIST digit dataset which is used for image classification.

To run the script, open the terminal and navigate to the source folder where the script is present.

```
python hyperparameter_tuning_using_optuna.py
```

Now, we need to launch the optuna dashboard in different process (different terminal)

```
optuna-dashboard sqlite:///db.sqlite3
```

![Alt text](/images/optuna-dashboard.mov?raw=true "Image Classification using Optuna")

2. <ins>Human in the loop optimization</ins> --> HITL_in_GANs.py
Human-in-the-loop (HITL) is a concept where humans play a role in machine learning or artificial intelligence systems. In HITL optimization in particular, humans are part of the optimization process.

The above script contains the HITL optimisation on MNIST digit dataset which is used for image generation using GANs.

To run the script, open the terminal and navigate to the source folder where the script is present.

```
python HITL_in_GANs.py
```

Now, we need to launch the optuna dashboard in different process (different terminal)

```
optuna-dashboard sqlite:///db.sqlite3 --artifact-dir ./artifact
```

![Alt text](/images/start.png?raw=true "Human in the loop optimization study created for digit classification")
![Alt text](/images/optimization.png?raw=true "Trials for respective HITL study")
