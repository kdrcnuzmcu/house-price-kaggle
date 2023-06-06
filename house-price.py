import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from typing import Optional

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr",  False)

train_ = pd.read_csv("House-Price/train.csv")
test_ = pd.read_csv("House-Price/test.csv")
sample_submission = pd.read_csv("House-Price/sample_submission.csv")

train = train_.copy()
test = test_.copy()

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

cat_cols, cat_but_car, num_cols = grab_col_names(train)

train.head()


train = train_.copy()
test = test_.copy()

RS = RobustScaler()
#MSSubClass
train["MSSubClass"] = RS.fit_transform(train[["MSSubClass"]])
test["MSSubClass"] = RS.fit_transform(test[["MSSubClass"]])

#MSZoning
temp = categorical_value_counts(train, "MSZoning", "SalePrice")
cats = temp[temp["Ratio"] < 0.08]["MSZoning"]
train.loc[train["MSZoning"].isin(cats), "MSZoning"] = "RareCat"
test.loc[test["MSZoning"].isin(cats), "MSZoning"] = "RareCat"

#LotFrontage
train["LotFrontage"] = train["LotFrontage"].fillna(0)
train["LotFrontage"] = RS.fit_transform(train[["LotFrontage"]])
test["LotFrontage"] = test["LotFrontage"].fillna(0)
test["LotFrontage"] = RS.fit_transform(test[["LotFrontage"]])

#LotArea
train["LotArea"] = RS.fit_transform(train[["LotArea"]])
test["LotArea"] = RS.fit_transform(test[["LotArea"]])

#Street

#Alley
train["Alley"] = train["Alley"].fillna("None")
test["Alley"] = test["Alley"].fillna("None")
temp = categorical_value_counts(train, "Alley", "SalePrice")
cats = temp[temp["Ratio"] < 0.05]["Alley"]
train.loc[train["Alley"].isin(cats), "Alley"] = "RareCat"
test.loc[test["Alley"].isin(cats), "Alley"] = "RareCat"

#LotShape
temp = categorical_value_counts(train, "LotShape", "SalePrice")
cats = temp[temp["Ratio"] < 0.05]["LotShape"]
train.loc[train["LotShape"].isin(cats), "LotShape"] = "RareCat"
test.loc[test["LotShape"].isin(cats), "LotShape"] = "RareCat"

#LandContour
temp = categorical_value_counts(train, "LandContour", "SalePrice")
cats = temp[temp["Ratio"] < 0.05]["LandContour"]
train.loc[train["LandContour"].isin(cats), "LandContour"] = "RareCat"
test.loc[test["LandContour"].isin(cats), "LandContour"] = "RareCat"

#Utilities

#LotConfig
temp = categorical_value_counts(train, "LotConfig", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["LotConfig"]
train.loc[train["LotConfig"].isin(cats), "LotConfig"] = "RareCat"
test.loc[test["LotConfig"].isin(cats), "LotConfig"] = "RareCat"

#LandScope
temp = categorical_value_counts(train, "LandSlope", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["LandSlope"]
train.loc[train["LandSlope"].isin(cats), "LandSlope"] = "RareCat"
test.loc[test["LandSlope"].isin(cats), "LandSlope"] = "RareCat"

#Neighborhood

#Condition1
temp = categorical_value_counts(train, "Condition1", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["Condition1"]
train.loc[train["Condition1"].isin(cats), "Condition1"] = "RareCat"
test.loc[test["Condition1"].isin(cats), "Condition1"] = "RareCat"

#Condition2
temp = categorical_value_counts(train, "Condition2", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["Condition2"]
train.loc[train["Condition2"].isin(cats), "Condition2"] = "RareCat"
test.loc[test["Condition2"].isin(cats), "Condition2"] = "RareCat"

#BldgType
temp = categorical_value_counts(train, "BldgType", "SalePrice")
cats = temp[temp["Ratio"] < 0.05]["BldgType"]
train.loc[train["BldgType"].isin(cats), "BldgType"] = "RareCat"
test.loc[test["BldgType"].isin(cats), "BldgType"] = "RareCat"

#HouseStyle
temp = categorical_value_counts(train, "HouseStyle", "SalePrice")
cats = temp[temp["Ratio"] < 0.03]["HouseStyle"]
train.loc[train["HouseStyle"].isin(cats), "HouseStyle"] = "RareCat"
test.loc[test["HouseStyle"].isin(cats), "HouseStyle"] = "RareCat"

#OverallQual

#RoofStyle
temp = categorical_value_counts(train, "RoofStyle", "SalePrice")
cats = temp[temp["Ratio"] < 0.03]["RoofStyle"]
train.loc[train["RoofStyle"].isin(cats), "RoofStyle"] = "RareCat"
test.loc[test["RoofStyle"].isin(cats), "RoofStyle"] = "RareCat"

#RoofMatl
temp = categorical_value_counts(train, "RoofMatl", "SalePrice")
cats = temp[temp["Ratio"] < 0.03]["RoofMatl"]
train.loc[train["RoofMatl"].isin(cats), "RoofMatl"] = "RareCat"
test.loc[test["RoofMatl"].isin(cats), "RoofMatl"] = "RareCat"

#Exterior1st
temp = categorical_value_counts(train, "Exterior1st", "SalePrice")
cats = temp[temp["Ratio"] < 0.01]["Exterior1st"]
train.loc[train["Exterior1st"].isin(cats), "Exterior1st"] = "RareCat"
test.loc[test["Exterior1st"].isin(cats), "Exterior1st"] = "RareCat"

#Exterior2nd
temp = categorical_value_counts(train, "Exterior2nd", "SalePrice")
cats = temp[temp["Ratio"] < 0.01]["Exterior2nd"]
train.loc[train["Exterior2nd"].isin(cats), "Exterior2nd"] = "RareCat"
test.loc[test["Exterior2nd"].isin(cats), "Exterior2nd"] = "RareCat"

#MasVnrType
train["MasVnrType"] = train["MasVnrType"].fillna("None")
test["MasVnrType"] = test["MasVnrType"].fillna("None")

#ExterQual
temp = categorical_value_counts(train, "ExterQual", "SalePrice")
cats = temp[temp["Ratio"] < 0.04]["ExterQual"]
train.loc[train["ExterQual"].isin(cats), "ExterQual"] = "RareCat"
test.loc[test["ExterQual"].isin(cats), "ExterQual"] = "RareCat"

#ExterCond
temp = categorical_value_counts(train, "ExterCond", "SalePrice")
cats = temp[temp["Ratio"] < 0.02]["ExterCond"]
train.loc[train["ExterCond"].isin(cats), "ExterCond"] = "RareCat"
test.loc[test["ExterCond"].isin(cats), "ExterCond"] = "RareCat"

#Foundation
temp = categorical_value_counts(train, "Foundation", "SalePrice")
cats = temp[temp["Ratio"] < 0.02]["Foundation"]
train.loc[train["Foundation"].isin(cats), "Foundation"] = "RareCat"
test.loc[test["Foundation"].isin(cats), "Foundation"] = "RareCat"

#BsmtQual
train["BsmtQual"] = train["BsmtQual"].fillna("None")
test["BsmtQual"] = test["BsmtQual"].fillna("None")

#BsmtCond
train["BsmtCond"] = train["BsmtCond"].fillna("None")
test["BsmtCond"] = test["BsmtCond"].fillna("None")

#BsmtExposure
train["BsmtExposure"] = train["BsmtExposure"].fillna("None")
test["BsmtExposure"] = test["BsmtExposure"].fillna("None")

#BsmtFinType1
train["BsmtFinType1"] = train["BsmtFinType1"].fillna("None")
test["BsmtFinType1"] = test["BsmtFinType1"].fillna("None")

#BsmtFinType2
train["BsmtFinType2"] = train["BsmtFinType2"].fillna("None")
test["BsmtFinType2"] = test["BsmtFinType2"].fillna("None")

#Heating
temp = categorical_value_counts(train, "Heating", "SalePrice")
cats = temp[temp["Ratio"] < 0.02]["Heating"]
train.loc[train["Heating"].isin(cats), "Heating"] = "RareCat"
test.loc[test["Heating"].isin(cats), "Heating"] = "RareCat"

#HeatingQC
temp = categorical_value_counts(train, "HeatingQC", "SalePrice")
cats = temp[temp["Ratio"] < 0.04]["HeatingQC"]
train.loc[train["HeatingQC"].isin(cats), "HeatingQC"] = "RareCat"
test.loc[test["HeatingQC"].isin(cats), "HeatingQC"] = "RareCat"

#Electrical
temp = categorical_value_counts(train, "Electrical", "SalePrice")
cats = temp[temp["Ratio"] < 0.04]["Electrical"]
train.loc[train["Electrical"].isin(cats), "Electrical"] = "RareCat"
test.loc[test["Electrical"].isin(cats), "Electrical"] = "RareCat"

#LowQualFinSF
train["LowQualFinSF"] = train["LowQualFinSF"].apply(lambda x: 0 if x == 0 else 1)
test["LowQualFinSF"] = test["LowQualFinSF"].apply(lambda x: 0 if x == 0 else 1)

#Functional
temp = categorical_value_counts(train, "Functional", "SalePrice")
cats = temp[temp["Ratio"] < 0.04]["Functional"]
train.loc[train["Functional"].isin(cats), "Functional"] = "RareCat"
test.loc[test["Functional"].isin(cats), "Functional"] = "RareCat"

#FireplaceQu
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
test["FireplaceQu"] = test["FireplaceQu"].fillna("None")
temp = categorical_value_counts(train, "FireplaceQu", "SalePrice")
cats = temp[temp["Ratio"] < 0.04]["FireplaceQu"]
train.loc[train["FireplaceQu"].isin(cats), "FireplaceQu"] = "RareCat"
test.loc[test["FireplaceQu"].isin(cats), "FireplaceQu"] = "RareCat"

#GarageType
train["GarageType"] = train["GarageType"].fillna("None")
test["GarageType"] = test["GarageType"].fillna("None")
temp = categorical_value_counts(train, "GarageType", "SalePrice")
cats = temp[temp["Ratio"] < 0.02]["GarageType"]
train.loc[train["GarageType"].isin(cats), "GarageType"] = "RareCat"
test.loc[test["GarageType"].isin(cats), "GarageType"] = "RareCat"

#GarageYrBlt
train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
test["GarageYrBlt"] = test["GarageYrBlt"].fillna(0)

#GarageFinish
train["GarageFinish"] = train["GarageFinish"].fillna("None")
test["GarageFinish"] = test["GarageFinish"].fillna("None")

#GarageQual
train["GarageQual"] = train["GarageQual"].fillna("None")
test["GarageQual"] = test["GarageQual"].fillna("None")
temp = categorical_value_counts(train, "GarageQual", "SalePrice")
cats = temp[temp["Ratio"] < 0.04]["GarageQual"]
train.loc[train["GarageQual"].isin(cats), "GarageQual"] = "RareCat"
test.loc[test["GarageQual"].isin(cats), "GarageQual"] = "RareCat"

#GarageCond
train["GarageCond"] = train["GarageCond"].fillna("None")
test["GarageCond"] = test["GarageCond"].fillna("None")
temp = categorical_value_counts(train, "GarageCond", "SalePrice")
cats = temp[temp["Ratio"] < 0.03]["GarageCond"]
train.loc[train["GarageCond"].isin(cats), "GarageCond"] = "RareCat"
test.loc[test["GarageCond"].isin(cats), "GarageCond"] = "RareCat"

#PavedDrive
temp = categorical_value_counts(train, "PavedDrive", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["PavedDrive"]
train.loc[train["PavedDrive"].isin(cats), "PavedDrive"] = "RareCat"
test.loc[test["PavedDrive"].isin(cats), "PavedDrive"] = "RareCat"

#EnclosedPorch
train["EnclosedPorch"] = train["EnclosedPorch"].apply(lambda x: 0 if x == 0 else 1)
test["EnclosedPorch"] = test["EnclosedPorch"].apply(lambda x: 0 if x == 0 else 1)

#3SsnPorch
train["3SsnPorch"] = train["3SsnPorch"].apply(lambda x: 0 if x == 0 else 1)
test["3SsnPorch"] = test["3SsnPorch"].apply(lambda x: 0 if x == 0 else 1)

#ScreenPorch
train["ScreenPorch"] = train["ScreenPorch"].apply(lambda x: 0 if x == 0 else 1)
test["ScreenPorch"] = test["ScreenPorch"].apply(lambda x: 0 if x == 0 else 1)

#PoolQC
train["PoolQC"] = train["PoolQC"].fillna("None")
test["PoolQC"] = test["PoolQC"].fillna("None")
temp = categorical_value_counts(train, "PoolQC", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["PoolQC"]
train.loc[train["PoolQC"].isin(cats), "PoolQC"] = "RareCat"
test.loc[test["PoolQC"].isin(cats), "PoolQC"] = "RareCat"

#Fence
train["Fence"] = train["Fence"].fillna("None")
test["Fence"] = test["Fence"].fillna("None")
temp = categorical_value_counts(train, "Fence", "SalePrice")
cats = temp[temp["Ratio"] < 0.07]["Fence"]
train.loc[train["Fence"].isin(cats), "Fence"] = "RareCat"
test.loc[test["Fence"].isin(cats), "Fence"] = "RareCat"

#MiscFeature
train["MiscFeature"] = train["MiscFeature"].fillna("None")
test["MiscFeature"] = test["MiscFeature"].fillna("None")
temp = categorical_value_counts(train, "MiscFeature", "SalePrice")
cats = temp[temp["Ratio"] < 0.01]["MiscFeature"]
train.loc[train["MiscFeature"].isin(cats), "MiscFeature"] = "RareCat"
test.loc[test["MiscFeature"].isin(cats), "MiscFeature"] = "RareCat"

#SaleType
temp = categorical_value_counts(train, "SaleType", "SalePrice")
cats = temp[temp["Ratio"] < 0.01]["SaleType"]
train.loc[train["SaleType"].isin(cats), "SaleType"] = "RareCat"
test.loc[test["SaleType"].isin(cats), "SaleType"] = "RareCat"

#SaleCondition
temp = categorical_value_counts(train, "SaleCondition", "SalePrice")
cats = temp[temp["Ratio"] < 0.01]["SaleCondition"]
train.loc[train["SaleCondition"].isin(cats), "SaleCondition"] = "RareCat"
test.loc[test["SaleCondition"].isin(cats), "SaleCondition"] = "RareCat"

X = train[["MSSubClass", "MSZoning", "LotFrontage", "LotArea",
           "Street", "Alley", "LotShape", "LandContour",
           "Utilities", "LotConfig", "LandSlope", "Neighborhood",
           "Condition1", "Condition2", "BldgType", "HouseStyle",
           "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
           "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
           "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond",
           "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
           "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2",
           "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC",
           "CentralAir", "Electrical", "1stFlrSF", "2ndFlrSF",
           "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
           "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
           "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces",
           "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish",
           "GarageCars", "GarageArea", "GarageQual", "GarageCond",
           "PavedDrive", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
           "3SsnPorch", "ScreenPorch", "PoolArea", "PoolQC",
           "Fence", "MiscFeature", "MiscVal", "MoSold",
           "YrSold", "SaleType", "SaleCondition"]]
X = pd.get_dummies(X, drop_first=True)

y = train["SalePrice"]

X.shape
X.head()

CB_model = CatBoostRegressor(verbose=False)
CB_model.fit(X, y)
print("Train Score:", CB_model.score(X, y))
y_pred = CB_model.predict(X)
print("MAPE:", mean_absolute_percentage_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

