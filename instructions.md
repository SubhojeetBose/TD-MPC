## Python environment installation instructions
- Make sure you have conda installed in your system. [Instructions link here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
- Then, get the `conda_env.yml` file, and from the same directory, run `conda env create -f conda_env.yml`. 
- Activate the environment - `conda activate ddrl_a3`
- Go into environment directory - `cd particle-envs`
- Install the environment - `pip install -e .`

## Running the code
- Make sure you have the environment activated, and you are in the `policy` directory.
- Command for Question 1: `python train_rl.py reward_type=dense agent=rl experiment=rl`
- Command for Question 2: `python train_bcrl.py reward_type=dense agent=bcrl experiment=bcrl`
