# Characterizing Concept Unlearning


## Development and Usage
This codebase uses `python3.10`. 
### Installation
This library is developed using [Poetry](https://python-poetry.org/), evidenced by the `pyproject.toml`. However, it can be installed either through Poetry or with `pip` + your favorite virtual environment.

#### Installation Using a Virtual Environment [Tested and Supported]
1. Create a virtual environment `python3.10 -m venv venv`
2. Source this environment `source venv/bin/activate`
3. From the base of the codebase, run `pip install -e .`   

#### Installation Using Poetry [Not supported by authors]
While `poetry` is used to manage the dependencies, and the authors use poetry, tests are run using `venv` and so the authors only commit to supporting installation using `virtualenv` or `venv`

1. Install Poetry(`curl -sSL https://install.python-poetry.org | python3 -`)
2. Navigate to the base of the codebase.
3. Run `poetry shell`
4. Run `poetry install`



Both of these methods for installation installs scripts for executing `private-pca`, and puts them on your `PYTHONPATH`. Running `poetry shell` or `source venv/bin/activate` will shell into the virtual environments with the code installed, and will allow you to run the executables directly.




