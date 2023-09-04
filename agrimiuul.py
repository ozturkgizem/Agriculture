##############################
### SETTING ENVIRONMENT
##############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import check_df, grab_col_names, correlation_matrix, outlier_thresholds, plot_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
#warnings.simplefilter('ignore')

##############################
### EXPLORATORY DATA ANALYSIS
##############################
df = pd.read_csv("Crop_recommendation.csv")
df.dtypes

df.head()

check_df(df)

##############################
#FEATURE ENGINEERING & DATA PREPARATION
##############################

def crop_recommendation_data_prep(dataframe):

    ##1: CROP SUITABILITY INDICES

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    def crop_suitability_indices(crop, col):
        low_limit, up_limit = outlier_thresholds(dataframe[dataframe["label"] == crop], col)
        dataframe.loc[(df["label"] == crop) & (dataframe[col] >= low_limit) & (dataframe[col] <= up_limit), f"NEW_{col.upper()}_OPTIMAL"] = 1
        dataframe.loc[(df["label"] == crop) & (dataframe[col] < low_limit) | (dataframe[col] > up_limit), f"NEW_{col.upper()}_OPTIMAL"] = 0

    for crop in df["label"].unique():
        for col in num_cols:
            crop_suitability_indices(crop, col)

    ##2: NUTRIENT BALANCE INDEX (NBI)

    dataframe["N/K"]= dataframe["N"] / dataframe["K"]
    dataframe["N/P"] = dataframe["N"] / dataframe["P"]
    dataframe["P/K"] = dataframe["P"] / dataframe["K"]

    ratios = ["N/K","N/P","P/K"]
    dataframe[ratios] = MinMaxScaler().fit_transform(dataframe[ratios])

    dataframe["NEW_NBI"] = np.sqrt(dataframe["N/K"] * dataframe["N/P"] * dataframe["P/K"])
    dataframe.drop(ratios, axis=1, inplace=True)


    ##3: CATEGORIZE BY PH LEVELS

    dataframe.loc[dataframe["ph"] < 7, "NEW_PH_CAT"] = "acidic"
    dataframe.loc[dataframe["ph"] > 7, "NEW_PH_CAT"] = "alkaline"
    dataframe.loc[dataframe["ph"] == 7, "NEW_PH_CAT"] = "neutral"
    dataframe.loc[(dataframe["ph"] >= 5.5) & (dataframe["ph"] <= 6.5), "NEW_PH_CAT"] = "optimal"

    ##4: CATEGORIZE BY RAINFALL
    #sns.catplot(data=df, x='label', y="rainfall", kind='box', height=10, aspect=22/8)
    #plt.title(f"{col}", size=12)
    #plt.xticks(rotation='vertical')
    #plt.show()

    #df["rainfall"].describe([0.10,0.20,0.30,0.33,0.40,0.50,0.60,0.66,0.70,0.75,0.80,0.90,0.99])

    dataframe["NEW_RAINFALL_CAT"] = pd.cut(x=dataframe["rainfall"], bins=[0, 44, 71, 111, 188, 300], labels=["extreme_low", "low", "medium", "high", "extreme_high"])

    ##5. ONE HOT ENCODER
    def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
        dff = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
        return dff

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dff = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(dff)

    X_scaled = StandardScaler().fit_transform(dff[num_cols])
    dff[num_cols] = pd.DataFrame(X_scaled, columns=dff[num_cols].columns)

    y = pd.DataFrame(dff["label"])
    X = dff.drop(["label"], axis=1)

    return X, y

X, y = crop_recommendation_data_prep(df)

##############################
#BASE MODELS
##############################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]
    stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Use stratified cross-validation

    for name, classifier in classifiers:
        cv_results = cross_val_score(classifier, X, y, cv=stratified_cv, scoring=scoring)
        print(f"{scoring}: {round(np.mean(cv_results), 4)} ({name}) ")

base_models(X,y)


