# latte
Data analysis of latent representations.


## Getting started with the code

Clone the repository:

```bash
   $ git clone git@github.com:BoevaLab/CanSig.git
   $ cd CanSig
```

As a developer, you will use some tools increasing code quality. You can install them by running

```bash
   $ pip install -r requirements-dev.txt
   $ pre-commit install
```

We use [black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest/) to lint the code, [pytype](https://github.com/google/pytype) to check whether the types agree, and [pytest](https://docs.pytest.org) to unit test the code.
These code quality checks are applied to every Pull Request via a workflow in the `.github/workflows` directory.




