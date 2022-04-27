# Modify the frequency domain of images

A playground repo.

## Usage

Setup:

1. Create a virtualenv: `python3.9 -m venv .venv39`
2. Activate it: `source .venv39/bin/activate`
3. Install dependencies in the virtualenv: `redo requirements/install`
   -- or if you don't have `redo`, you can instead:
    1. Install pip-sync: `pip install -r requirements/pip-tools.lst`
    2. Install the rest of the dependencies: `pip-sync requirements/dev.lst`

To use the script, make sure the virtualenv is active (as above) and
run `python -m fimg <in-path> <out-path> <etc...>`.

...but you'll probably have to modify the script to do what you want!
It doesn't have a full CLI.