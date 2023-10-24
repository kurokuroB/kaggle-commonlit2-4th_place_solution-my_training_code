# kaggle-commonlit2-4th_place_solution-my_training_code

My training code which is used in 4th place solution(team) in kaggle「[CommonLit - Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)」.

The solution of our team is [here](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524).

My training code is about model no1 and no6 in this solution.

## Content

./  
┗ training  
　 ┗ 238.py : training code for model no1(deberta-v3-large).  
　 ┗ 259.py : training code for model no6(deberta-v3-large-squad2).

## How to train

### 1. Perform the following code in terminal.

```
git clone https://github.com/kurokuroB/kaggle-commonlit2-4th_place_solution-my_training_code
cd kaggle-commonlit2-4th_place_solution-my_training_code
mkdir data
cd data
mkdir input
mkdir output
```

### 2. Put the competition's data into data/input folder.

The necessary data is [here](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data)

### 3. Train models

You can train models when perform the following code in terminal.

```
cd ..
python training/238.py
python training/259.py
```

After training, model weights will be put into data/output folder.

- File names
  - model1.ckpt : the weight of model no1
  - model6.ckpt : the weight of model no6

If there are missing libraries in your environment, please install them based on the `requirements.txt`.
