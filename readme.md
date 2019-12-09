# NBA winnder prediction
## dataset
The data set is downloaded from [basketball-reference.com](https://www.basketball-reference.com/). Choose `Seasons` and pick a season such as `2018-19`. Get `Team Per Game Stats`, `Opponent Per Game Stats` and `Miscellaneous Stats`, and save them as `cvs` file in `data` subpath.
Just like 
 
 ![data files](./images/data_file.png)
 
then the features can be create using those files. And match result label information can be got from `Schedule and Results`(as the same as studima200). Just save all results into `csv` file.
Please check './data/Year_2016_2017.csv', './data/Year_2017_2018.csv' and './data/Year_2018_2019.csv'.
check `./data` subpath is like:

![all_files](./images/all_files.png)
 
## model
Machine Learning method: Tecision Tree, Random Forest, XGBoost, Logistic Regression and Naive Bayes (Gaussian) are programmed in `studima200Project.py`. 
~~~python
python ./studima200Project.py
~~~


## requirements.txt
sklearn
pandas
numpy
xgboost