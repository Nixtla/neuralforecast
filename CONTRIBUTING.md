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

#### Fork and clone the repository

1. **Fork the repository**: Navigate to the [neuralforecast repository](https://github.com/Nixtla/neuralforecast) and click the "Fork" button in the top-right corner. This creates a copy of the repository under your GitHub account.

2. **Clone your fork**: Clone your forked repository to your local machine (replace `YOUR_USERNAME` with your GitHub username):

   * HTTPS: `git clone https://github.com/YOUR_USERNAME/neuralforecast.git`
   * SSH: `git clone git@github.com:YOUR_USERNAME/neuralforecast.git`
   * GitHub CLI: `gh repo clone YOUR_USERNAME/neuralforecast`

3. **Navigate to the project directory**:

   ```bash
   cd neuralforecast
   ```

4. **Add the upstream remote** (Optional - skip if you prefer using GitHub's web UI): Add the original Nixtla repository as an upstream remote to keep your fork in sync:

   ```bash
   git remote add upstream https://github.com/Nixtla/neuralforecast.git
   ```

   > **Note**: This step is not required if you use GitHub's "Sync fork" button in the web UI to keep your fork updated. However, configuring the upstream remote is recommended for a smoother local development workflow.

5. **Verify your remotes**: Confirm that you have both `origin` (your fork) and `upstream` (original repo) configured:

   ```bash
   git remote -v
   ```

6. **Keep your fork up to date**: Before starting work on a new feature or fix, sync your fork with the upstream repository:

   **Option A - Using Git locally** (requires step 4):
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

   **Option B - Using GitHub's web UI**: Navigate to your fork on GitHub and click the "Sync fork" button, then pull the changes locally:
   ```bash
   git checkout main
   git pull origin main
   ```

7. **Create a feature branch**: Always create a new branch for your work:

   ```bash
   git checkout -b your-feature-branch-name
   ```

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

#### Submitting your changes

After making your changes and testing them:

1. **Commit your changes**: Make sure your commits are clear and descriptive:

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

2. **Push to your fork**: Push your feature branch to your forked repository:

   ```bash
   git push origin your-feature-branch-name
   ```

3. **Create a Pull Request**: Go to the [original neuralforecast repository](https://github.com/Nixtla/neuralforecast) and you should see a prompt to create a pull request from your recently pushed branch. Click on "Compare & pull request" and fill in the PR details following the guidelines above.

4. **Keep your PR updated**: If you need to make changes based on review feedback, commit and push to the same branch:

   ```bash
   git add .
   git commit -m "Address review comments"
   git push origin your-feature-branch-name
   ```

   The PR will automatically update with your new commits.

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
uv pip install -e '.[dev, docs]' lazydocs
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
