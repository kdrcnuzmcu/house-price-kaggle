import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from qbstyles import mpl_style

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from typing import Optional

# from ModuleWizard.helpers import HelperFunctions

# pandasOptions = PandasOptions()
# pandasOptions.PrintOptions()
# pandasOptions.SetOptions(1, 4)

import pandas as pd
train_ = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\House-Price\train.csv")
test_ = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\House-Price\test.csv")
sample_submission = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\House-Price\sample_submission.csv")

class HelperFunctions():
    from typing import Optional
    def __init__(self, dataframe):
        self.dataframe = dataframe
    def QuickView(self):
        print(f"""
Rows: {self.dataframe.shape[0]}
Columns: {self.dataframe.shape[1]}
* * *

HEAD:
{self.dataframe.head()}
* * *

TAIL:
{self.dataframe.tail()}
* * *

SAMPLES:
{self.dataframe.sample(5)}
        """)
    def Variables(self):
        print(f"""
{self.dataframe.info()} 
* * * 
      
NUMBER OF NULLS:
{self.dataframe.isnull().sum()}        
* * * 
  
NUMBER OF UNIQUES:
{self.dataframe.nunique()}   
* * *   
        """)
    def GrabColNames(self, cat_th=10, car_th=20, verbose=False):
        cat_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in self.dataframe.columns if self.dataframe[col].nunique() < cat_th and self.dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in self.dataframe.columns if self.dataframe[col].nunique() > car_th and self.dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # Numerical Columns
        num_cols = [col for col in self.dataframe.columns if self.dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        # Results
        if verbose:
            print(f"Observations: {self.dataframe.shape[0]}")
            print(f"Variables: {self.dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')

        return cat_cols, cat_but_car, num_cols
    def CategoricalsByTarget(self, col, target, rare: Optional[float] = None):
        temp = self.dataframe.groupby(col, dropna=False).agg(Count=(col, lambda x: x.isnull().count()), \
                                                        Ratio=(col, lambda x: x.isnull().count() / len(self.dataframe)), \
                                                        Target_Ratio=(target, lambda x: x.sum() / self.dataframe[target].sum())) \
                                                    .sort_values("Count", ascending=False).reset_index()
        if rare is not None:
            rares = temp.loc[temp["Ratio"] <= float(rare), col].tolist()
            self.dataframe.loc[self.dataframe[col].isin(rares), col] = "Rare Category"
            print("---- Done! --- ")
            print(self.dataframe.groupby(col).agg(Count=(col, lambda x: x.count()), \
                                      Ratio=(col, lambda x: x.count() / len(self.dataframe)), \
                                      Target_Ratio=(target, lambda x: x.sum() / self.dataframe[target].sum())) \
                  .sort_values("Count", ascending=False).reset_index(), "\n")
        else:
            print(temp, "\n")
    def Outliers(self, col, low_Quantile=0.25, high_Quantile=0.75, adjust=False):
        Q1 = self.dataframe[col].quantile(low_Quantile)
        Q3 = self.dataframe[col].quantile(high_Quantile)
        IQR = Q3 - Q1
        low_Limit = Q1 - (1.5 * IQR)
        up_Limit = Q3 + (1.5 * IQR)

        if len(self.dataframe[self.dataframe[col] > up_Limit]) > 0:
            print(col, ": Higher Outlier!")
        if len(self.dataframe[self.dataframe[col] < low_Limit]) > 0:
            print(col, ": Lower Outlier!")

        if adjust:
            self.dataframe.loc[(self.dataframe[col] < low_Limit), col] = low_Limit
            self.dataframe.loc[(self.dataframe[col] > up_Limit), col] = up_Limit
            print(f"{col}: Done!")
    def ExtractFromDatetime(self, col):
        self.dataframe[col] = pd.to_datetime(self.dataframe[col])
        self.dataframe["YEAR"] = self.dataframe[col].dt.year
        self.dataframe["MONTH"] = self.dataframe[col].dt.month
        self.dataframe["DAY"] = self.dataframe[col].dt.day
        self.dataframe["DAYOFWEEK"] = self.dataframe[col].dt.dayofweek + 1
        self.dataframe["HOUR"] = self.dataframe[col].dt.hour
        self.dataframe["WEEK"] = self.dataframe[col].dt.isocalendar().week

from ModuleWizard.module_wizard import PandasOptions
PandasOptions().PrintOptions()
PandasOptions().SetOptions(1, 4, 2)

HelperFunctions(train_).QuickView()
HelperFunctions(train_).Variables()
HelperFunctions(train_).CategoricalsByTarget("GarageType", "SalePrice")
HelperFunctions(train_).Outliers("LotArea")

train_.info

#region Fonksiyonlar
def grab_col_names(df, cat_th=10, car_th=20):

    # Categorical Columns
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical Columns
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # Results
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

def categorical_value_counts(df, col, target: None, rare: Optional[float] = None):
    temp = df.groupby(col, dropna=False).agg(Count=(col, lambda x: x.isnull().count()), \
                               Ratio=(col, lambda x: x.isnull().count() / len(df)), \
                               Target_Ratio=(target, lambda x: x.sum() / df[target].sum())) \
        .sort_values("Count", ascending=False).reset_index()

    if rare is not None:
        rares = temp.loc[temp["Ratio"] <= float(rare), col].tolist()
        df.loc[df[col].isin(rares), col] = "Rare Category"
        print("---- Done! --- ")
        temp = df.groupby(col).agg(Count=(col, lambda x: x.isnull().count()), \
                                  Ratio=(col, lambda x: x.count() / len(df)), \
                                  Target_Ratio=(target, lambda x: x.sum() / df[target].sum())) \
              .sort_values("Count", ascending=False).reset_index()
    return temp

def outliers(df, col, low_Quantile=0.25, high_Quantile=0.75, adjust=False):
    Q1 = df[col].quantile(low_Quantile)
    Q3 = df[col].quantile(high_Quantile)
    IQR = Q3 - Q1
    low_Limit = Q1 - (1.5 * IQR)
    up_Limit = Q3 + (1.5 * IQR)

    if len(df[df[col] > up_Limit]) > 0:
        print(col, ": Higher Outlier!")
    if len(df[df[col] < low_Limit]) > 0:
        print(col, ": Lower Outlier!")

    if adjust:
        df.loc[(df[col] < low_Limit), col] = low_Limit
        df.loc[(df[col] > up_Limit), col] = up_Limit
        print(col, ": Done!")

def on_isleme(col, navalue=None, rare=None, scale=None):
    if train[col].dtype == "int64":
        train[col] = train[col].fillna(navalue)
        test[col] = train[col].fillna(navalue)
        if scale == "binary":
            train[col] = train[col].apply(lambda x: 0 if x == 0 else 1)
            test[col] = test[col].apply(lambda x: 0 if x == 0 else 1)
        else:
            train[col] = scale.fit_transform(train[[col]])
            test[col] = scale.fit_transform(test[[col]])
    elif train[col].dtype == "O":
        train[col] = train[col].fillna(navalue)
        test[col] = test[col].fillna(navalue)
        temp = categorical_value_counts(train, col, "SalePrice")
        cats = temp[temp["Ratio"] < rare][col]
        train.loc[train[col].isin(cats), col] = "RareCat"
        test.loc[test[col].isin(cats), col] = "RareCat"

def plot_importance(model, features, num = 3):
    mpl_style(dark=True)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(num="{}".format(model.__class__.__name__), figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.get_current_fig_manager()
    plt.show()

def model_fit(estimator, X, y, tag="Train"):
    print("# * ~  -- * -- --{}-- -- * -- ~ * #".format(estimator.__class__.__name__))
    model = estimator.fit(X, y)
    print("Train Score:", model.score(X, y))
    y_pred = model.predict(X)
    print("# * ~  -- --{} Dataset-- -- ~ * #".format(tag))
    metriks(y, y_pred)

def metriks(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))

#endregion

train = train_.copy()
test = test_.copy()
cat_cols, cat_but_car, num_cols = grab_col_names(train)
RS = RobustScaler()

#region Preprocessing
on_isleme("MSSubClass", navalue=0, scale=RS) #MSSubClass
on_isleme("MSZoning", "None", 0.08) #MSZoning
on_isleme("LotFrontage", 0, scale=RS) #LotFrontage
on_isleme("LotArea", navalue=0, scale=RS) #LotArea
on_isleme("Alley", "None", 0.05) #Alley
on_isleme("LotShape", "None", 0.05) #LotShape
on_isleme("LandContour", "None", 0.05) #LandContour
on_isleme("LotConfig", "None", 0.07) #LotConfig
on_isleme("LandSlope", "None", 0.07) #LandScope
on_isleme("Condition1", "None", 0.07) #Condition1
on_isleme("Condition2", "None", 0.07) #Condition2
on_isleme("BldgType", "None", 0.05) #BldgType
on_isleme("HouseStyle", "None", 0.03) #HouseStyle
on_isleme("RoofStyle", "None", 0.03) #RoofStyle
on_isleme("RoofMatl", "None", 0.03) #RoofMatl
on_isleme("Exterior1st", "None", 0.01) #Exterior1st
on_isleme("Exterior2nd", "None", 0.01) #Exterior2nd
on_isleme("MasVnrType", "None") #MasVnrType
on_isleme("ExterQual", "None", 0.04) #ExterQual
on_isleme("ExterCond", "None", 0.02) #ExterCond
on_isleme("Foundation", "None", 0.02) #Foundation
on_isleme("BsmtQual", "None") #BsmtQual
on_isleme("BsmtCond", "None") #BsmtCond
on_isleme("BsmtExposure", "None") #BsmtExposure
on_isleme("BsmtFinType1", "None") #BsmtFinType1
on_isleme("BsmtFinType2", "None") #BsmtFinType2
on_isleme("Heating", "None", 0.02) #Heating
on_isleme("HeatingQC", "None", 0.04) #HeatingQC
on_isleme("Electrical", "None", 0.04) #Electrical
on_isleme("Functional", "None", 0.04) #Functional
on_isleme("FireplaceQu", "None", 0.04) #FireplaceQu
on_isleme("GarageType", "None", 0.02) #GarageType
on_isleme("GarageYrBlt", 0) #GarageYrBlt
on_isleme("GarageFinish", "None") #GarageFinish
on_isleme("GarageQual", "None", 0.04) #GarageQual
on_isleme("GarageCond", "None", 0.03) #GarageCond
on_isleme("PavedDrive", "None", 0.07) #PavedDrive
on_isleme("PoolQC", "None", 0.07) #PoolQC
on_isleme("Fence", "None", 0.07) #Fence
on_isleme("MiscFeature", "None", 0.01) #MiscFeature
on_isleme("SaleType", "None", 0.01) #SaleType
on_isleme("SaleCondition", "None", 0.01) #SaleCondition
on_isleme("LowQualFinSF", 0,  scale="binary") #LowQualFinSF
on_isleme("EnclosedPorch", 0,  scale="binary") #EnclosedPorch
on_isleme("3SsnPorch", 0, scale="binary") #3SsnPorch
on_isleme("ScreenPorch", 0, scale="binary") #ScreenPorch

cats = ["MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street",
       "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
       "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType",
       "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
       "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
       "MasVnrArea", "ExterQual", "ExterCond", "Foundation", "BsmtQual",
       "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2",
       "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC",
       "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
       "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
       "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional",
       "Fireplaces", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish",
       "GarageCars", "GarageArea", "GarageQual", "GarageCond", "PavedDrive",
       "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
       "PoolArea", "PoolQC", "Fence", "MiscFeature", "MiscVal",
       "MoSold", "YrSold", "SaleType", "SaleCondition"]

X_train = train[cats]
X_test = train[cats]
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
y_train = train["SalePrice"]
#endregion

#region Preprocessing2
train2 = train_[["GarageArea", "SalePrice"]]
X = train2[["GarageArea"]]
y = train2["SalePrice"]

X["GarageLabels"] = pd.cut(X["GarageArea"], bins=[-1, 1, 350, 800, 1000, 1450], labels=[0, 1, 2, 3, 4])
X = pd.get_dummies(data=X, columns=["GarageLabels"], drop_first=True)
LGBM_model = LGBMRegressor()
model_fit(LGBM_model, X, y)
feature_imp = pd.DataFrame({"Value": LGBM_model.feature_importances_, "Feature": X.columns}).sort_values("Value", ascending=False)

#endregion

#region EDA

def kesif(col):
    print(train_[col].head(10))
    print(train_[col].describe())
    print(train_[col].isnull().sum(), "/", train_.shape[0])

kesif("GarageArea")
col = "LotFrontage"
train_[train_[col] < 80]["SalePrice"].mean()
train_[train_[col] > 80]["SalePrice"].mean()

kesif("GarageArea")
train_["GarageArea"].plot.density(color="green")
sns.histplot(X["GarageArea"], kde=True, bins=140)
#plt.hist(train_["GarageArea"], bins=140, density=True)
plt.show()
#endregion

#region Model Fitting
print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("CatBoost Raw Dataset"))
CB_model = CatBoostRegressor(verbose=False)
model_fit(CB_model, X_train, y_train)

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("LightGBM Raw Dataset"))
LGBM_model = LGBMRegressor()
model_fit(LGBM_model, X_train, y_train)

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("XGBoost Raw Dataset"))
XGB_model = XGBRegressor()
model_fit(XGB_model, X_train, y_train)

# Raw Dataset - Feature Importance Plot
plot_importance(CB_model, X_train, num=25)

plot_importance(LGBM_model, X_train, num=25)
feature_imp = pd.DataFrame({"Value": LGBM_model.feature_importances_, "Feature": X_train.columns}).sort_values("Value", ascending=False)
feature_imp.head()

plot_importance(XGB_model, X_train, num=25)
#endregion

#region Train-Test-Split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("CatBoost Train-Test-Split"))
CB_model = CatBoostRegressor(verbose=False)
model_fit(CB_model, Xtrain, ytrain)
print("### ### ###")
print("Train Score:", CB_model.score(Xtest, ytest))
y_pred = CB_model.predict(Xtest)
metriks(ytest, y_pred)

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("LightGBM Train-Test-Split"))
LGBM_model = LGBMRegressor()
model_fit(LGBM_model, Xtrain, ytrain)
print("### ### ###")
print("Train Score:", LGBM_model.score(Xtest, ytest))
y_pred = LGBM_model.predict(Xtest)
metriks(ytest, y_pred)

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("XGBoost Train-Test-Split"))
XGB_model = XGBRegressor()
model_fit(XGB_model, Xtrain, ytrain)
print("### ### ###")
print("Train Score:", XGB_model.score(Xtest, ytest))
y_pred = LGBM_model.predict(Xtest)
metriks(ytest, y_pred)

# Train-Test-Split - Feature Importance Plot
plot_importance(CB_model, Xtrain, num=25)

plot_importance(LGBM_model, Xtrain, num=25)

plot_importance(XGB_model, Xtrain, num=25)
#endregion Train-Test-Split

#region CrossValidation
# Cross Validation - Raw Dataset
print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("CatBoost Cross Validation"))
CB_model = CatBoostRegressor(verbose=False)
print(np.mean(cross_val_score(CB_model, X_train, y_train, cv=5)))

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("LightGBM Cross Validation"))
LGBM_model = LGBMRegressor()
print(np.mean(cross_val_score(LGBM_model, X_train, y_train, cv=5)))

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("XGBoost Cross Validation"))
XGB_model = XGBRegressor()
print(np.mean(cross_val_score(XGB_model, X_train, y_train, cv=5)))

# Cross Validation - Train-Test-Split
print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("CatBoost Cross Validation"))
CB_model = CatBoostRegressor(verbose=False)
print(np.mean(cross_val_score(CB_model, Xtrain, ytrain, cv=5)))

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("LightGBM Cross Validation"))
LGBM_model = LGBMRegressor()
print(np.mean(cross_val_score(LGBM_model, Xtrain, ytrain, cv=5)))

print("# * ~ -- -- -- -- -- --{}-- -- -- -- -- -- ~ * #".format("XGBoost Cross Validation"))
XGB_model = XGBRegressor()
print(np.mean(cross_val_score(XGB_model, Xtrain, ytrain, cv=5)))
#endregion

#region Export Submission

# Raw Dataset
y_pred_test = XGB_model.predict(X_test)
y_pred_test = pd.Series(y_pred_test)

# Train-Test-Split
y_pred_test = XGB_model.predict(Xtest)
y_pred_test = pd.Series(y_pred_test)

sample_submission.loc[:, "SalePrice"] = y_pred_test
sample_submission.to_csv("preds/pred5.csv", index=False)
#endregion
