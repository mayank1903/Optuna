# Optuna
This repo contains all the experiments related to Optuna library in python

__Experiments folder consists of all the .ipynb notebooks related to Optuna and its understanding__

1. <ins>Human in the loop optimization</ins> --> human_in_the_loop_optimisation.py
Human-in-the-loop (HITL) is a concept where humans play a role in machine learning or artificial intelligence systems. In HITL optimization in particular, humans are part of the optimization process.

The above script contains the HITL optimisation on MNIST digit dataset.

To run the script, open the terminal and navigate to the source folder where the script is present.

```
python human_in_the_loop_optimisation.py
```

Now, we need to launch the optuna dashboard in different process (different terminal)

```
optuna-dashboard sqlite:///db.sqlite3 --artifact-dir ./artifact
```

![Alt text](/images/start.png?raw=true "Human in the loop optimization study created for digit classification")
![Alt text](/images/optimization.png?raw=true "Trials for respective HITL study")
