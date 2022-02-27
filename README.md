# Overview and Testing Details
These repository contains an implementation for an Orthogonal Tucker ALS algorithm,
and then tests it on face image data and my own data.
To use and test this implementation on your own machine,
take the following steps:

1. `git clone https://github.com/zblanks/tucker.git` to your desired location
2. Ensure you have Python and Poetry installed on your machine
    * See https://python-poetry.org/docs/ for instructions on how to get Poetry installed
3. Navigate to the directory containing the Tucker repository
4. `poetry install`
    * This will build the dependencies 
    and install the tucker package in the virtual environment
5. `poetry shell`
    * This spawns a .venv shell allowing you to run this experiment
6. To run the various experiments
type: `python experiments.py`

And that's it! I summarized my experimental details in the submitted homework,
but feel free to play around with this implementation on your own.