# Project 1: Image Filtering and Hybrid Images

## Setup

1. Install Miniconda. (If you already have Miniconda installed, you can skip this step)
2. Create a conda environment using the appropriate terminal and command.

- On Windows, open the installed "Anaconda Powershell Prompt".
- On MacOS and Linux, you can open a terminal window.
- Modify and run the command in the terminal, replace the “<OS>” in the following
  command with your OS (Linux, Mac, Windows): `conda env create -f proj1_configs/proj1_env_<OS>.yml`

3. Check if the cv_proj1 environment has been created properly.

- Run: `conda env list`

4. Activate the conda environment.

- Run: `conda activate cv_proj1`

- To deactivate it, run: `conda deactivate`

5. Install the project packages.

- Run: `pip install -e` . inside the repo folder.
- This should be unnecessary for Project 1 but is good practice when setting up a new
  conda environment that may have pip requirements.

6. Open the jupyter notebook to work on the project.

- Run: `jupyter notebook ./proj1_code/proj1.ipynb`

## Testing & Submission

1. Ensure that all sanity checks are passing

- Run: `pytest proj1_unit_tests` inside the proj1_code folder.

2. Compress your code into a zip for submission

- Run: `python zip_submission.py --gt_username <your_gt_username>`s

3. Submit the zip to Gradescope for the code part

- NOTE: we have two separate assignments on Gradescope for Project 1
  - (4476) Project 1 – 70pts
  - (6476) Project 1 – 80pts
- If you are a 4476 student
  - If you don’t do the extra credit part, submit to the (4476) assignment
  - If you do the extra credit part, submit to the (6476) assignment
- If you are a 6476 student, submit to the (6476) assignment
- 5pts will be deducted if you make submission to the wrong assignment

4. Save the PowerPoint as PDF and submit the PDF to Gradescope for the report part
