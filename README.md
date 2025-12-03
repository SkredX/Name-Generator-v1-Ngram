# Simple Bigram Model: Markov Chains

This repository contains a minimal character-level bigram Markov model for generating realistic name-like text. The notebook `n gram-name-generator.ipynb` demonstrates how character transitions are counted, normalized into probabilities, and sampled to generate new names. The model operates using simple statistical dependencies between consecutive characters and provides an interpretable introduction to generative text modeling.

## Repository Structure

Simple-bigram-model--Markov-chains/
Bigram Model/


n gram-name-generator.ipynb


name generator.py


names.txt


Markov_README.txt


## How to Run the Model

### 1. Using the Notebook
Open the file `Bigram Model/n gram-name-generator.ipynb` in Jupyter Notebook or JupyterLab.  
Run the cells from top to bottom. The notebook will:
- load `names.txt`
- compute bigram counts and conditional probabilities
- generate new names based on the learned Markov transitions

### 2. Using the Python Script
From a terminal, navigate into the `Bigram Model` folder:
`cd "Bigram Model"`

Run the generator:
`python "name generator.py"`


The script will:
- load the `names.txt` dataset
- build the bigram transition model
- prompt the user for starting letters and the number of names to generate
- output sampled names from the trained distribution

Ensure that `names.txt` remains in the same folder as the script or notebook.

## Credits
This project is based on concepts and explanations taught by Andrej Karpathy, whose work on character-level language models and educational code walkthroughs inspired this implementation.
