# -*- coding:utf-8 -*-
"""
NBA winner prediction by machine learning method:
    Logistic Regression, Naive Bayes (Gaussian), Random Forest, Decision Tree, XGBoost model

project from: https://www.freelancer.cn/projects/python/NBA-winner-prediction/details

2019/12/9   Jie Y.      Init

"""

import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
folder = 'data'


# calculate Elo score
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif 2100 <= winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


# read csv file and build information matrix
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')

    # print(team_stats1.info())
    return team_stats1.set_index('Team', inplace=False, drop=True)


def get_elo(team):
    try:
        return team_elos[team]
    except:
        # if not, init elo score as base_elo
        team_elos[team] = base_elo
        return team_elos[team]


def build_dataSet(team_stats, all_data):
    # print("Building data set..")
    for index, row in all_data.iterrows():
        WLoc = ''
        if float(row['PTS']) > float(row['PTS.1']):
            Wteam = row['Visitor/Neutral']
            Lteam = row['Home/Neutral']
            WLoc = 'V'  # winner team is Visitor team
        else:
            Wteam = row['Home/Neutral']
            Lteam = row['Visitor/Neutral']
            WLoc = 'H'  # winner team is home team

        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        if WLoc == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        team1_features = [team1_elo]
        team2_features = [team2_elo]

        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)

        if WLoc == 'H':
            X.append(team1_features + team2_features)
            y.append(1)
        else:
            X.append(team2_features + team1_features)
            y.append(0)

        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), np.array(y)


if __name__ == '__main__':
    print('2017-18 season Stats are chosen as feature, target information is from 2018-19 season.')
    Mstat = pd.read_csv(folder + '/17-18Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/17-18Opponent_Per_Game_Stat.csv')
    Tstat = pd.read_csv(folder + '/17-18Team_Per_Game_Stat.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/Year_2018_2019.csv')
    X, y = build_dataSet(team_stats, result_data)
    # data sets should be shuffled.
    per = np.random.permutation(len(y))
    X = X[per, :]
    y = y[per]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('sample of trainning set:{},sample of test set:{}'.format(len(y_train), len(y_test)))

    # build and train LogisticRegression model
    lr = LogisticRegression(penalty='l2', tol=0.00001, C=0.5, solver='liblinear')
    lr.fit(X_train, y_train)
    score_lr = lr.score(X_test, y_test)
    score_lr_train = lr.score(X_train, y_train)
    print('score of Logistic Regression model in test set is {:.4f} (score of training set:{:.4f})'.format(score_lr,
                                                                                                           score_lr_train))

    # build and train GaussianNB model
    # more setting:https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    gnb = GaussianNB(var_smoothing=1e-09)
    gnb.fit(X_train, y_train)
    score_gnb = gnb.score(X_test, y_test)
    score_gnb_train = gnb.score(X_train, y_train)
    print('score of Naive Bayes (Gaussian) model in test set is {:.4f} (score of training set:{:.4f})'.format(score_gnb,
                                                                                                              score_gnb_train))

    # build and train Random Forest model
    # more setting:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    score_rf = rf.score(X_test, y_test)
    score_rf_train = rf.score(X_train, y_train)
    print('score of Random Forest model in test set is {:.4f} (score of training set:{:.4f})'.format(score_rf,
                                                                                                     score_rf_train))

    # build and train Decision Tree model
    # more setting: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    score_dt = dt.score(X_test, y_test)
    score_dt_train = dt.score(X_train, y_train)
    print('score of Decision Tree model in test set is {:.4f} (score of training set:{:.4f})'.format(score_dt,
                                                                                                     score_dt_train))

    # build and train XGBoost model
    xgb = XGBClassifier(learning_rate=0.01, max_depth=5)
    xgb.fit(X_train, y_train)
    score_xgb = xgb.score(X_test, y_test)
    score_xgb_train = xgb.score(X_train, y_train)
    print('score of XGBoost model in test set is {:.4f} (score of training set:{:.4f})'.format(score_xgb,
                                                                                               score_xgb_train))
