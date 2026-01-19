# Project description
The purpose of this project is to train a transformer-based model for next-token prediction. We will be using a pre-trained model, possibly from Hugging Face, and fine-tuning it on the danish-foundation-models/danish-gigaword dataset using the classic unsupervised “masking” training paradigm. The Danish Gigaword dataset, developed by the IT University of Denmark, contains over a billion words. The dataset is open source and can be downloaded from Huggingface. We don’t think data version control is neccesary for our project, because we don’t expect the datasets to change. However, we might still try to implement it, for example, using DVC, just to get a feel of how it works in practice.

Having trained our model, we can use the model to generate arbitrary Danish sentences starting from a given prompt. By prompt, we mean the start of some sentence which the model will then, hopefully, be able to finish. Given the time, we also plan to implement additional datasets to train on as much data as possible.

We will use a combination of Pytorch Lightning, Hydra, and Weights & Biases to train, configure, and monitor the performance of our models.  We plan to implement as many methods from the course as possible, since we already have a good grasp of how these three frameworks work. We will also try to set up an API for doing remote inference. This way, users can prompt our model with any sentence that they would like to get the model to finish.

We start by fine-tuning a lightweight transformer model in order to validate the data pipeline. We subsequently fine-tune Mistral-7B using LoRA to get optimal performance, and test our setup on a heavier transformer model.

We aim to implement a complete continuous integration pipeline with automated testing and verification in a containerized environment to minimize deployment friction. The project will apply standard DevOPS workflows common in larger projects. New feature pull request are required to pass all automated tests and be reviewed by two team members before being merged with the main branch.

* **Nikolaj**: `s214653@dtu.dk`
* **Gabriel**: `s214615@dtu.dk`
* **Gustav**: `s214657@dtu.dk`
* **Andreas**: `s214630@dtu.dk`

# Instructions
## Clone repo
```bash
git clone https://github.com/kommodeskab/mlops-hestnet.git
```
Remember that you have to login. This can be done very easily with the Github CLI. Look it up.

## New branch
It is FORBIDDEN (literally) to push to main. Therefore, when you want to make a new change, make a new branch, for example like this:
```bash
git branch -b <new-branch-name>
```
You can make all the changes you want on this branch, then commit and push.

## Make changes to main
As mentioned, you cannot push directly to main. Instead, push your changes to your own branch. If you want to merge these changes into main, then (after having pushed) go to the Github page for the Repo and find your branch. There should be a yellow/green banner saying "`<new-branch-name> had recent pushes`" with a button "Compare & pull request". Click that button, write a description of your changes, and create a pull request. After that, someone (maybe you) can review the changes and merge them into main. You can now see the changes on main:
```bash
git checkout main
git pull origin main
```
Optionally, if your own branch was made to make a new specific change, you can delete your branch both locally and on Github:
```bash
git branch -d <new-branch-name>     # delete locally
git push origin --delete <new-branch-name> # delete on Github
```
Otherwise, you can choose to "reset" or update your branch as described below.

## "Reset" you branch
Have your branced diverged from main? i.e. there have been some changes on main that you would like to also have on your branch? Then type:
```bash
git checkout <new-branch-name>
git merge main # or 'git rebase main'
```
Alternatively, if you want to throw out all your changes and forget them (note: all your changes on the branch will be lost):
```bash
git checkout <new-branch-name>
git reset --hard origin/main
```

## Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Initialize Environment
To build the environment, run:
```bash
uvx invoke build
```
This runs the function `build.py` as seen in `tasks.py`. It is probably a good idea to inspect the functions in `tasks.py` before running them.
Also remember to activate the environment after building it:
```bash
source .venv/bin/activate
```
Almost all commands assume that the environment is activated.

## Add dependency (package)
If you want to add a new dependency (package) to the environment, use:
```bash
uv add <package-name>
```

## WandB
Log into WandB by pasting the API key into the .env file.
The API key can be found at: [wandb.ai/authorize](https://wandb.ai/authorize)

## Check formatting
To check (and automatically fix some) code formatting, use:
```bash
uvx invoke format
```

## Check typing
To check typing using mypy, use:
```bash
uvx invoke typing --filename <path-to-file-or-directory>
```
You can omit the `--filename` argument to check the entire codebase.

## Run an experiment
To run an experiment, use:
```bash
python main.py +experiment=<config-file-name>
```
You can add additional overrides as needed using Hydra. For example, to change the batch size, use:
```bash
python main.py +experiment=<config-file-name> data.batch_size=64
```

## Testing a model
You can also test a model after having trained it. You use the same config file but change `phase` to `test` and specify the `id` of the run. As default, the last checkpoint (`last.ckpt`) will be used. You can override this by specifying `ckpt_path` in the config. This can be done from the command line as well. For example:
```bash
python main.py +experiment=<config-file-name> phase=test continue_from_id=<run-id>
```
Or if you want to specify a custom checkpoint path:
```bash
python main.py +experiment=<config-file-name> phase=test continue_from_id=<run-id> ckpt_filename=<filename-of-checkpoint>
```

## Run the dummy example
You can try out the dummy example to get started by running:
```bash
python main.py +experiment=dummy
```

## Pytest and coverage
To run the tests using pytest, use:
```bash
pytest
```
Test a specific file:
```bash
pytest <path-to-file>
```

Run a specific test function in a file:
```bash
pytest <path-to-file>::<test-function-name>
```

To run with print statements enabled, use:
```bash
pytest -s
```

To see how much of the code is covered by tests, use:
```bash
coverage run -m pytest
coverage report -m
```

## Pre-commit hooks
To install pre-commit hooks, run:
```bash
uv run pre-commit install
```
This will automatically run formatting and typing checks before each commit. You can see in the file `.pre-commit-config.yaml` which checks are enabled. You can manually run all pre-commit hooks on all files whenever you want by running:
```bash
uv run pre-commit run --all-files
```
You can also commit without the pre-commit hooks (if they are annoying you and you just really want to push) by using:
```bash
git commit --no-verify
```

## Sweeps
Sweep configuration files are located in `configs/sweeps/`. To initialize a new sweep, use:
```bash
uvx invoke buildsweep --name <sweep-name>
```
Where `<sweep-name>` corresponds to the name of the YAML file in `configs/sweeps/` (without the `.yaml` extension). See `configs/sweeps/dummy.yaml` for an example of a sweep configuration file.
This will output a sweep ID that can be used to start agents. To start an agent for the sweep, use:
```bash
wandb agent <sweep-id>
```
The sweep id also includes the entity and project name, for example:
```bash
wandb agent kommodeskab-danmarks-tekniske-universitet-dtu/sweeps/vbh4iehv
```
You can do this locally or on a remote machine. If you want to submit this sweep to a remote HPC cluster, you can for example use:
```bash
uvx invoke submit --command='wandb agent <sweep-id>' --job-name='wandb-sweep-<sweep-name>' --time='02:00' --gpus=1 --cpus=4 --mem=4
```
