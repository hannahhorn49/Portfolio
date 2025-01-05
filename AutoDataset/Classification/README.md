# README

## To Setup (One Time Only)
`git clone (your github repo url here)`  
`cd` into your local repo you just cloned 

## Options for Editing Jupyter Notebooks
* VS Code with the Jupyter Extension
* Jupyter Lab
    * Start your Docker container and run the following command in its bash shell: `jupyter lab --ip='0.0.0.0' --port=8888 --no-browser --allow-root --notebook-dir=/home`

## To Run Unit Tests
Run `pytest --verbose`
* This command runs all the discovered tests in the project
* You can run individual test modules with
    * `pytest --verbose test_myclassifiers.py`
* Note: the `-s` flag can be helpful because it will show print statement output from test execution

## What not to Modify
You may not modify:
* Anything in the `test/` directory (if it exists)
* Any hidden files (e.g. files/folders that start with a `.`)

Note: you will need to modify `test_myclassifiers.py` to complete the assignment
