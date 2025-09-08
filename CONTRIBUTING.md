# How to contribute

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

## Do you have a feature request?

* Ensure that it hasn't been yet implemented in the `main` branch of the repository and that there's not an Issue requesting it yet.
* Open a new issue and make sure to describe it clearly, mention how it improves the project and why its useful.

## Do you want to fix a bug or implement a feature?

Bug fixes and features are added through pull requests (PRs).

## PR submission guidelines

* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Ensure that your PR includes a test that fails without your patch, and passes with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely get rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.

### Local setup for working on a PR

#### Clone the repository

* HTTPS: `git clone https://github.com/Nixtla/neuralforecast.git`
* SSH: `git clone git@github.com:Nixtla/neuralforecast.git`
* GitHub CLI: `gh repo clone Nixtla/neuralforecast`

#### Set up an environment

Create a virtual environment to install the library's dependencies. We recommend [astral's uv](https://github.com/astral-sh/uv).
Once you've created the virtual environment you should activate it and then install the library in editable mode along with its development dependencies.

Install `uv` and create a virtual environment:

```bash
pip install uv
uv venv --python 3.11
```

Then, activate the virtual environment:

* On Linux/MacOS:

```bash
source .venv/bin/activate
```

* On Windows:

```bash
.\.venv\Scripts\activate
```

Now, install the library. Make sure to specify the desired [PyTorch backend](https://docs.astral.sh/uv/reference/cli/#uv-pip-install--torch-backend):

```bash
uv pip install -e ".[dev]" --torch-backend auto # uv will decide the optimal backend automatically
uv pip install -e ".[dev]" --torch-backend cpu # for cpu backend
uv pip install -e ".[dev]" --torch-backend cu118 # for CUDA 11.8 PyTorch backend
```

You can install other optional dependencies using

```sh
uv pip install -e ".[dev,aws,spark]"
```

#### Install pre-commit hooks

```sh
pre-commit install
pre-commit run --files neuralforecast/*
```

#### Running tests

To run the tests, run

```sh
uv run pytest
```

#### Viewing documentation locally

The new documentation pipeline relies on `quarto`, `mintlify` and `lazydocs`.

#### install quarto

Install `quarto` from &rarr; [this link](https://quarto.org/docs/get-started/)

#### install mintlify

> [!NOTE]
> Please install Node.js before proceeding.

```sh
npm i -g mint
```

For additional instructions, you can read about it &rarr; [this link](https://mintlify.com/docs/installation).

```sh
uv pip install -e '.[dev]' lazydocs
make all_docs
```

Finally to view the documentation

```sh
make preview_docs
```

## Do you want to contribute to the documentation?

* The docs are automatically generated from the docstrings in the `neuralforecast` folder.
* To contribute, ensure your docstrings follow the Google style format.
* Once your docstring is correctly written, the documentation framework will scrape it and regenerate the corresponding `.mdx` files and your changes will then appear in the updated docs.
* To contribute, examples/how-to-guides, make sure you submit clean notebooks, with cleared formatted LaTeX, links and images.
* Make an appropriate entry in the `docs/mintlify/mint.json` file.
* Run `make all_docs` to regenerate the documentation.
* Run `make preview_docs` to view and test the documentation locally.

### Example google-style docstring

```py
def function_name(parameter1, parameter2):
    """Brief summary of the function's purpose.

    Detailed explanation of what the function does, its behavior, and any
    important considerations. This section can span multiple lines.

    Args:
        parameter1 (type): Description of parameter1.
        parameter2 (type): Description of parameter2.

    Returns:
        type: Description of the return value.

    Raises:
        ExceptionType: Description of the circumstances under which this
                       exception is raised.

    Example:
        Examples can be provided to demonstrate how to use the function.
        Literal blocks can be included for code snippets::

            result = function_name(10, 'hello')
            print(result)

    Notes:
        Any additional notes or important information.
    """
    # Function implementation
    pass

```
