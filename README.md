# latte
Data analysis of latent representations.


## Getting started with the code

Clone the repository:

```bash
$ git clone git@github.com:BoevaLab/latte.git
$ cd latte
```

At this point it may be beneficial to create a new [Python virtual environment](https://docs.python.org/3.8/tutorial/venv.html). There are multiple solutions for this step, including [Miniconda](https://docs.conda.io/en/latest/miniconda.html). We aim at Python 3.8 version and above.

Then you install the package _in editable mode_, together with the testing utilities:

```bash
$ pip install -e ".[test]"
  
```

### Development tools

As a developer, you will use some tools increasing code quality. You can install them by running

```bash
   $ pip install -r requirements-dev.txt
   $ pre-commit install
```

We use [black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest/) to lint the code, [pytype](https://github.com/google/pytype) to check whether the types agree, and [pytest](https://docs.pytest.org) to unit test the code.
These code quality checks are applied to every Pull Request via a workflow in the `.github/workflows` directory.

