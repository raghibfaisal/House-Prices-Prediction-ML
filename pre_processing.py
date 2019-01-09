import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load the training dataset and test dataset into dataframes train_set and test_set respectively.

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
target = np.log(train_set.SalePrice)
"""
Outliers were noticed in Garage Area (Garage Area > 1200), GrLivArea, TotalBsmtSf, 1stFlrSF.
Removing outliers in below commands.
"""
train_set=train_set[train_set['GarageArea']<1200]   #Outliers for garage area removed
train_set=train_set[train_set['GrLivArea']<4500]    #Outliers for garage liv area removed
train_set=train_set[train_set['TotalBsmtSF']<6000]  #Outlier for total sqft basement area removed
train_set=train_set[train_set['1stFlrSF']<4500]     #Outlier for First floor sqft removed
#sns.scatterplot('GrLivArea',target,data=train_set)
#plt.show()

"""
Dealing with categorical columns. One hot encoding.
"""

#MSZoning
#print(train_set.MSZoning.value_counts(),'\n')

def msz_encode1(x):
    if x == 'RL':
        return 1
    else:
        return 0

def msz_encode2(x):
    if x == 'RM':
        return 1
    else:
        return 0

train_set['RL_MSZ']=train_set.MSZoning.apply(msz_encode1)
train_set['RM_MSZ']=train_set.MSZoning.apply(msz_encode2)
#for test set
test_set['RL_MSZ']=test_set.MSZoning.apply(msz_encode1)
test_set['RM_MSZ']=test_set.MSZoning.apply(msz_encode2)

train_set.drop(['MSZoning'],axis=1,inplace=True)
test_set.drop(['MSZoning'],axis=1,inplace=True)


#Street

train_set['Pave_Street']=pd.get_dummies(train_set.Street,drop_first=True)
test_set['Pave_Street']=pd.get_dummies(test_set.Street,drop_first=True)

train_set.drop(['Street'],axis=1,inplace=True)
test_set.drop(['Street'],axis=1,inplace=True)

#Alley
"""
For the Alley feature we can go ahead and drop the whole column as we have only 91 values available (6%).
"""
train_set.drop(['Alley'],axis=1,inplace=True)
test_set.drop(['Alley'],axis=1,inplace=True)

#LotShape
def ls_encode1(x):
    if x == 'Reg':
        return 1
    else:
        return 0

train_set['Reg_LotShape']=train_set.LotShape.apply(ls_encode1)
#for test set
test_set['Reg_LotShape']=test_set.LotShape.apply(ls_encode1)

train_set.drop(['LotShape'],axis=1,inplace=True)
test_set.drop(['LotShape'],axis=1,inplace=True)

#LandContour

def land_contour_encode(x):
    if x == 'Lvl':
        return 1
    else:
        return 0

train_set['Lvl_LandContour']=train_set.LandContour.apply(land_contour_encode)
#for test set
test_set['Lvl_LandContour']=test_set.LandContour.apply(land_contour_encode)

train_set.drop(['LandContour'],axis=1,inplace=True)
test_set.drop(['LandContour'],axis=1,inplace=True)

#Utilities
"""
We can drop this as all values are same (AllPub). Hence, it's not going to affect our model much.
"""
train_set.drop(['Utilities'],axis=1,inplace=True)
test_set.drop(['Utilities'],axis=1,inplace=True)

#LotConfig

def lot_config_encode1(x):
    if x == 'Inside':
        return 1
    else:
        return 0

def lot_config_encode2(x):
    if x == 'Corner':
        return 1
    else:
        return 0

train_set['Inside_LotConfig']=train_set.LotConfig.apply(lot_config_encode1)
train_set['Corner_LotConfig']=train_set.LotConfig.apply(lot_config_encode2)
#for test set
test_set['Inside_LotConfig']=test_set.LotConfig.apply(lot_config_encode1)
test_set['Corner_LotConfig']=test_set.LotConfig.apply(lot_config_encode2)

train_set.drop(['LotConfig'],axis=1,inplace=True)
test_set.drop(['LotConfig'],axis=1,inplace=True)


#LandSLope

def land_slope_encode(x):
    if x == 'Gtl':
        return 1
    else:
        return 0

train_set['Gtl_LandSlope']=train_set.LandSlope.apply(land_slope_encode)
#for test set
test_set['Gtl_LandSlope']=test_set.LandSlope.apply(land_slope_encode)

train_set.drop(['LandSlope'],axis=1,inplace=True)
test_set.drop(['LandSlope'],axis=1,inplace=True)


#Neighborhood
"""
nei=sns.barplot('Neighborhood','SalePrice',data=train_set)
for i in nei.get_xticklabels():
    i.set_rotation(45)

mean_nei = train_set.groupby(['Neighborhood'])['SalePrice'].mean()
print(mean_nei)
print('\nMax',mean_nei.max())
print('\nMin',mean_nei.min())

Since there are 26 values to the neighborhood feature we decided to group the different
neighborhoods into 4 groups namels - SE (super-expensive), E (expensive), A (Average)
and C (cheap).
"""

def neighbor_encode1(neib):

    if neib == 'StoneBr' or neib == 'NridgHt' or neib == 'NoRidge':
        return 1
    else:
        return 0

train_set['SE_Neighborhood'] = train_set.Neighborhood.apply(neighbor_encode1)

test_set['SE_Neighborhood'] = test_set.Neighborhood.apply(neighbor_encode1)


def neighbor_encode2(neib):

    if neib == 'Blmngtn' or neib == 'ClearCr' or neib == 'CollgCr' or neib == 'Crawfor'\
            or neib == 'Gilbert' or neib == 'NWAmes' or neib == 'SawyerW' or neib == 'Somerst'\
            or neib == 'Timber' or neib == 'Veenker':
        return 1
    else:
        return 0

train_set['E_Neighborhood'] = train_set.Neighborhood.apply(neighbor_encode2)

test_set['E_Neighborhood'] = test_set.Neighborhood.apply(neighbor_encode2)



def neighbor_encode3(neib):

    if neib == 'Blueste' or neib == 'BrkSide' or neib == 'Edwards' or neib == 'Mitchel'\
            or neib == 'NAmes' or neib == 'NPkVill' or neib == 'OldTown' or neib == 'SWISU'\
            or neib == 'Sawyer':
        return 1
    else:
        return 0

train_set['A_Neighborhood'] = train_set.Neighborhood.apply(neighbor_encode3)

test_set['A_Neighborhood'] = test_set.Neighborhood.apply(neighbor_encode3)

def neighbor_encode4(neib):

    if neib == 'BrDale' or neib == 'IDOTRR' or neib == 'MeadowV':
        return 1
    else:
        return 0

train_set['C_Neighborhood'] = train_set.Neighborhood.apply(neighbor_encode4)

test_set['C_Neighborhood'] = test_set.Neighborhood.apply(neighbor_encode4)

train_set.drop(['Neighborhood'],axis=1,inplace=True)
test_set.drop(['Neighborhood'],axis=1,inplace=True)


#Condition1

#sns.barplot('Condition1','SalePrice',data=train_set)
def condition1_encode(x):
    if x == 'Norm':
        return 1
    else:
        return 0

train_set['Normal_con1']=train_set.Condition1.apply(condition1_encode)
#for test set
test_set['Normal_con1']=test_set.Condition1.apply(condition1_encode)

train_set.drop(['Condition1'],axis=1,inplace=True)
test_set.drop(['Condition1'],axis=1,inplace=True)


#Condition2
"""
deleting this as all values almost same
"""
train_set.drop(['Condition2'],axis=1,inplace=True)
test_set.drop(['Condition2'],axis=1,inplace=True)

#BldgType

def bldgtype_encode(x):
    if x == '1Fam':
        return 1
    else:
        return 0

train_set['1Fam_btype']=train_set.BldgType.apply(bldgtype_encode)
#for test set
test_set['1Fam_btype']=test_set.BldgType.apply(bldgtype_encode)

train_set.drop(['BldgType'],axis=1,inplace=True)
test_set.drop(['BldgType'],axis=1,inplace=True)


#HouseStyle

def HouseStyle_encode1(x):
    if x == '1Story':
        return 1
    else:
        return 0

def HouseStyle_encode2(x):
    if x == '2Story':
        return 1
    else:
        return 0

train_set['1story_house_style']=train_set.HouseStyle.apply(HouseStyle_encode1)
train_set['2story_house_style']=train_set.HouseStyle.apply(HouseStyle_encode2)
#for test set
test_set['1story_house_style']=test_set.HouseStyle.apply(HouseStyle_encode1)
test_set['2story_house_style']=test_set.HouseStyle.apply(HouseStyle_encode2)

train_set.drop(['HouseStyle'],axis=1,inplace=True)
test_set.drop(['HouseStyle'],axis=1,inplace=True)


#RoofStyle

def RoofStyle_encode1(x):
    if x == 'Gable':
        return 1
    else:
        return 0

def RoofStyle_encode2(x):
    if x == 'Hip':
        return 1
    else:
        return 0

train_set['Gable_roof_style']=train_set.RoofStyle.apply(RoofStyle_encode1)
train_set['Hip_roof_style']=train_set.RoofStyle.apply(RoofStyle_encode2)
#for test set
test_set['Gable_roof_style']=test_set.RoofStyle.apply(RoofStyle_encode1)
test_set['Hip_roof_style']=test_set.RoofStyle.apply(RoofStyle_encode2)

train_set.drop(['RoofStyle'],axis=1,inplace=True)
test_set.drop(['RoofStyle'],axis=1,inplace=True)


#RoofMatl


def RoofMatl_encode(x):
    if x == 'CompShg':
        return 1
    else:
        return 0

train_set['Compshg_roof_mtl']=train_set.RoofMatl.apply(RoofMatl_encode)
#for test set
test_set['Compshg_roof_mtl']=test_set.RoofMatl.apply(RoofMatl_encode)

train_set.drop(['RoofMatl'],axis=1,inplace=True)
test_set.drop(['RoofMatl'],axis=1,inplace=True)


#Exterior1st
"""
We have divided the the various exterior types into 2 main groups - Exp_Exterior1st and Avg_Exterior1st.
The cheaper exteriors will be automatically figured out as others.
"""
#sns.barplot('Exterior1st','SalePrice',data=train_set)
def Exterior1st_encode1(x):
    if x == 'VinylSd' or x== 'CemntBd' or x == 'Stone' or x == 'ImStucc':
        return 1
    else:
        return 0

def Exterior1st_encode2(x):
    if x == 'MetalSd' or x == 'WdShing' or x == 'HdBoard' or x == 'BrkFace' or x == 'Wd Sdng'\
            or x == 'Plywood' or x == 'Stucco':
        return 1
    else:
        return 0

train_set['Exp_Exterior1st']=train_set.Exterior1st.apply(Exterior1st_encode1)
train_set['Avg_Exterior1st']=train_set.Exterior1st.apply(Exterior1st_encode2)
#for test set
test_set['Exp_Exterior1st']=test_set.Exterior1st.apply(Exterior1st_encode1)
test_set['Avg_Exterior1st']=test_set.Exterior1st.apply(Exterior1st_encode2)

train_set.drop(['Exterior1st'],axis=1,inplace=True)
test_set.drop(['Exterior1st'],axis=1,inplace=True)


#Exterior2nd

#sns.barplot('Exterior1st','SalePrice',data=train_set)
def Exterior2nd_encode1(x):
    if x == 'VinylSd' or x== 'CemntBd' or x == 'Stone' or x == 'ImStucc':
        return 1
    else:
        return 0

def Exterior2nd_encode2(x):
    if x == 'MetalSd' or x == 'WdShing' or x == 'HdBoard' or x == 'BrkFace' or x == 'Wd Sdng'\
            or x == 'Plywood' or x == 'Stucco':
        return 1
    else:
        return 0

train_set['Exp_Exterior2nd']=train_set.Exterior2nd.apply(Exterior2nd_encode1)
train_set['Avg_Exterior2nd']=train_set.Exterior2nd.apply(Exterior2nd_encode2)
#for test set
test_set['Exp_Exterior2nd']=test_set.Exterior2nd.apply(Exterior2nd_encode1)
test_set['Avg_Exterior2nd']=test_set.Exterior2nd.apply(Exterior2nd_encode2)

train_set.drop(['Exterior2nd'],axis=1,inplace=True)
test_set.drop(['Exterior2nd'],axis=1,inplace=True)


#MasVnrType

def MasVnrType_encode1(x):
    if x == 'None':
        return 1
    else:
        return 0

def MasVnrType_encode2(x):
    if x == 'BrkFace':
        return 1
    else:
        return 0

train_set['No_MasVnrType']=train_set.MasVnrType.apply(MasVnrType_encode1)
train_set['Brkface_MasVnrType']=train_set.MasVnrType.apply(MasVnrType_encode2)
#for test set
test_set['No_MasVnrType']=test_set.MasVnrType.apply(MasVnrType_encode1)
test_set['Brkface_MasVnrType']=test_set.MasVnrType.apply(MasVnrType_encode2)

train_set.drop(['MasVnrType'],axis=1,inplace=True)
test_set.drop(['MasVnrType'],axis=1,inplace=True)

#ExterQual

def ExterQual_encode1(x):
    if x == 'Ex' or x == 'Gd':
        return 1
    else:
        return 0

def ExterQual_encode2(x):
    if x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0

train_set['Good_ex_qual']=train_set.ExterQual.apply(ExterQual_encode1)
train_set['avd_ex_qual']=train_set.ExterQual.apply(ExterQual_encode2)
#for test set
test_set['Good_ex_qual']=test_set.ExterQual.apply(ExterQual_encode1)
test_set['avd_ex_qual']=test_set.ExterQual.apply(ExterQual_encode2)
                                                  
train_set.drop(['ExterQual'],axis=1,inplace=True)
test_set.drop(['ExterQual'],axis=1,inplace=True)

#ExterCond

def ExterCond_encode1(x):
    if x == 'Ex' or x == 'Gd':
        return 1
    else:
        return 0


def ExterCond_encode2(x):
    if x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0


train_set['Good_ex_cond'] = train_set.ExterCond.apply(ExterCond_encode1)
train_set['avd_ex_cond'] = train_set.ExterCond.apply(ExterCond_encode2)
# for test set
test_set['Good_ex_cond'] = test_set.ExterCond.apply(ExterCond_encode1)
test_set['avd_ex_cond'] = test_set.ExterCond.apply(ExterCond_encode2)

train_set.drop(['ExterCond'], axis=1, inplace=True)
test_set.drop(['ExterCond'], axis=1, inplace=True)


#Foundation

#sns.barplot('Foundation','SalePrice',data=train_set)
def Foundation_encode1(x):
    if x == 'PConc' or x == 'Wood':
        return 1
    else:
        return 0


def Foundation_encode2(x):
    if x == 'CBlock' or x == 'Stone' or x == 'BrkTil':
        return 1
    else:
        return 0


train_set['Exp_foundation'] = train_set.Foundation.apply(Foundation_encode1)
train_set['Avg_foundation'] = train_set.Foundation.apply(Foundation_encode2)
# for test set
test_set['Exp_foundation'] = test_set.Foundation.apply(Foundation_encode1)
test_set['Avg_foundation'] = test_set.Foundation.apply(Foundation_encode2)

train_set.drop(['Foundation'], axis=1, inplace=True)
test_set.drop(['Foundation'], axis=1, inplace=True)


#BsmtQual

def BsmtQual_encode1(x):
    if x == 'Ex' or x == 'Gd':
        return 1
    else:
        return 0


def BsmtQual_encode2(x):
    if x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0


train_set['Good_BsmtQual'] = train_set.BsmtQual.apply(BsmtQual_encode1)
train_set['avg_BsmtQual'] = train_set.BsmtQual.apply(BsmtQual_encode2)
# for test set
test_set['Good_BsmtQual'] = test_set.BsmtQual.apply(BsmtQual_encode1)
test_set['avg_BsmtQual'] = test_set.BsmtQual.apply(BsmtQual_encode2)

train_set.drop(['BsmtQual'], axis=1, inplace=True)
test_set.drop(['BsmtQual'], axis=1, inplace=True)

#BsmtCond

def BsmtCond_encode1(x):
    if x == 'Ex' or x == 'Gd':
        return 1
    else:
        return 0


def BsmtCond_encode2(x):
    if x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0


train_set['Good_BsmtCond'] = train_set.BsmtCond.apply(BsmtCond_encode1)
train_set['avg_BsmtCond'] = train_set.BsmtCond.apply(BsmtCond_encode2)
# for test set
test_set['Good_BsmtCond'] = test_set.BsmtCond.apply(BsmtCond_encode1)
test_set['avg_BsmtCond'] = test_set.BsmtCond.apply(BsmtCond_encode2)

train_set.drop(['BsmtCond'], axis=1, inplace=True)
test_set.drop(['BsmtCond'], axis=1, inplace=True)


#BsmtExposure
def BsmtExposure_encode1(x):
    if x == 'No':
        return 1
    else:
        return 0


def BsmtExposure_encode2(x):
    if x == 'Gd':
        return 1
    else:
        return 0


train_set['No_BsmtExposure'] = train_set.BsmtExposure.apply(BsmtExposure_encode1)
train_set['Gd_BsmtExposure'] = train_set.BsmtExposure.apply(BsmtExposure_encode2)
# for test set
test_set['No_BsmtExposure'] = test_set.BsmtExposure.apply(BsmtExposure_encode1)
test_set['Gd_BsmtExposure'] = test_set.BsmtExposure.apply(BsmtExposure_encode2)

train_set.drop(['BsmtExposure'], axis=1, inplace=True)
test_set.drop(['BsmtExposure'], axis=1, inplace=True)



#BsmtFinType1

#sns.barplot('BsmtFinType1','SalePrice',data=train_set)
def BsmtFinType1_encode1(x):
    if x == 'GLQ':
        return 1
    else:
        return 0

train_set['GLQ_BsmtFinType1']=train_set.BsmtFinType1.apply(BsmtFinType1_encode1)
#for test set
test_set['GLQ_BsmtFinType1']=test_set.BsmtFinType1.apply(BsmtFinType1_encode1)
train_set.drop(['BsmtFinType1'],axis=1,inplace=True)
test_set.drop(['BsmtFinType1'],axis=1,inplace=True)


#BsmtFinType2

train_set.drop(['BsmtFinType2'],axis=1,inplace=True)
test_set.drop(['BsmtFinType2'],axis=1,inplace=True)


#Heating

#sns.barplot('Heating','SalePrice',data=train_set)

def heating_encode(x):
    if x == 'GasA' or x == 'GasW':
        return 1
    else:
        return 0

train_set['Gas_heating'] = train_set.Heating.apply(heating_encode)
test_set['Gas_heating'] = test_set.Heating.apply(heating_encode)

train_set.drop(['Heating'],axis=1,inplace=True)
test_set.drop(['Heating'],axis=1,inplace=True)

#HeatingQC

#sns.barplot('HeatingQC','SalePrice',data=train_set)
def HeatingQC_encode1(x):
    if x == 'Ex':
        return 1
    else:
        return 0

def HeatingQC_encode2(x):
    if x == 'Gd' or x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0

train_set['Ex_heating'] = train_set.HeatingQC.apply(HeatingQC_encode1)
train_set['Gd_heating'] = train_set.HeatingQC.apply(HeatingQC_encode2)

test_set['Ex_heating'] = test_set.HeatingQC.apply(HeatingQC_encode1)
test_set['Gd_heating'] = test_set.HeatingQC.apply(HeatingQC_encode2)

train_set.drop(['HeatingQC'],axis=1,inplace=True)
test_set.drop(['HeatingQC'],axis=1,inplace=True)


#CentralAir

train_set['Central_air_y']=pd.get_dummies(train_set.CentralAir,drop_first=True)
test_set['Central_air_y']=pd.get_dummies(test_set.CentralAir,drop_first=True)

train_set.drop(['CentralAir'],axis=1,inplace=True)
test_set.drop(['CentralAir'],axis=1,inplace=True)

#Electrical

#sns.barplot('Electrical','SalePrice',data=train_set)
def electrical_encode(x):
    if x == 'SBrkr':
        return 1
    else:
        return 0

train_set['SBrkr_Electrical'] = train_set.Electrical.apply(electrical_encode)
test_set['SBrkr_Electrical'] = test_set.Electrical.apply(electrical_encode)

train_set.drop(['Electrical'],axis=1,inplace=True)
test_set.drop(['Electrical'],axis=1,inplace=True)

#KitchenQual
#sns.barplot('KitchenQual','SalePrice',data=train_set)

def KitchenQual_encode1(x):
    if x == 'Ex':
        return 1
    else:
        return 0

def KitchenQual_encode2(x):
    if x == 'Gd':
        return 1
    else:
        return 0

def KitchenQual_encode3(x):
    if x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0

train_set['Ex_KitchenQual'] = train_set.KitchenQual.apply(KitchenQual_encode1)
train_set['Gd_KitchenQual'] = train_set.KitchenQual.apply(KitchenQual_encode2)
train_set['Avg_KitchenQual'] = train_set.KitchenQual.apply(KitchenQual_encode3)

test_set['Ex_KitchenQual'] = test_set.KitchenQual.apply(KitchenQual_encode1)
test_set['Gd_KitchenQual'] = test_set.KitchenQual.apply(KitchenQual_encode2)
test_set['Avg_KitchenQual'] = test_set.KitchenQual.apply(KitchenQual_encode3)

train_set.drop(['KitchenQual'],axis=1,inplace=True)
test_set.drop(['KitchenQual'],axis=1,inplace=True)

#Functional

#sns.barplot('Functional','SalePrice',data=train_set)
def Functional_encode(x):
    if x == 'Typ':
        return 1
    else:
        return 0

train_set['Typ_Functional'] = train_set.Functional.apply(Functional_encode)
test_set['Typ_Functional'] = test_set.Functional.apply(Functional_encode)

train_set.drop(['Functional'],axis=1,inplace=True)
test_set.drop(['Functional'],axis=1,inplace=True)

#FireplaceQu
#sns.barplot('FireplaceQu','SalePrice',data=train_set)
def FireplaceQu_encode1(x):
    if x == 'Ex':
        return 1
    else:
        return 0

def FireplaceQu_encode2(x):
    if x == 'Gd' or x == 'TA' or x == 'Fa':
        return 1
    else:
        return 0

train_set['Ex_FireplaceQu'] = train_set.FireplaceQu.apply(FireplaceQu_encode1)
train_set['Gd_FireplaceQu'] = train_set.FireplaceQu.apply(FireplaceQu_encode2)

test_set['Ex_FireplaceQu'] = test_set.FireplaceQu.apply(FireplaceQu_encode1)
test_set['Gd_FireplaceQu'] = test_set.FireplaceQu.apply(FireplaceQu_encode2)

train_set.drop(['FireplaceQu'],axis=1,inplace=True)
test_set.drop(['FireplaceQu'],axis=1,inplace=True)

#GarageType
#sns.barplot('GarageType','SalePrice',data=train_set)
def GarageType_encode(x):
    if x == 'Attchd' or x == 'BuiltIn':
        return 1
    else:
        return 0

train_set['Ex_GarageType'] = train_set.GarageType.apply(GarageType_encode)
test_set['Ex_GarageType'] = test_set.GarageType.apply(GarageType_encode)

train_set.drop(['GarageType'],axis=1,inplace=True)
test_set.drop(['GarageType'],axis=1,inplace=True)

#GarageFinish
#sns.barplot('GarageFinish','SalePrice',data=train_set)
def GarageFinish_encode(x):
    if x == 'RFn' or x == 'Fin':
        return 1
    else:
        return 0

train_set['Fin_GarageFinish'] = train_set.GarageFinish.apply(GarageFinish_encode)
test_set['Fin_GarageFinish'] = test_set.GarageFinish.apply(GarageFinish_encode)

train_set.drop(['GarageFinish'],axis=1,inplace=True)
test_set.drop(['GarageFinish'],axis=1,inplace=True)

#GarageQual
#sns.barplot('GarageQual','SalePrice',data=train_set)
def GarageQual_encode1(x):
    if x == 'Ex' or x == 'Gd':
        return 1
    else:
        return 0

def GarageQual_encode2(x):
    if x == 'TA':
        return 1
    else:
        return 0

train_set['Ex_GarageQual'] = train_set.GarageQual.apply(GarageQual_encode1)
train_set['Avg_GarageQual'] = train_set.GarageQual.apply(GarageQual_encode2)

test_set['Ex_GarageQual'] = test_set.GarageQual.apply(GarageQual_encode1)
test_set['Avg_GarageQual'] = test_set.GarageQual.apply(GarageQual_encode2)

train_set.drop(['GarageQual'],axis=1,inplace=True)
test_set.drop(['GarageQual'],axis=1,inplace=True)

#GarageCond
#sns.barplot('GarageCond','SalePrice',data=train_set)
def GarageCond_encode1(x):
    if x == 'TA' or x == 'Gd':
        return 1
    else:
        return 0

train_set['Gd_GarageCond'] = train_set.GarageCond.apply(GarageCond_encode1)

test_set['Gd_GarageCond'] = test_set.GarageCond.apply(GarageCond_encode1)

train_set.drop(['GarageCond'],axis=1,inplace=True)
test_set.drop(['GarageCond'],axis=1,inplace=True)

#PavedDrive
#sns.barplot('PavedDrive','SalePrice',data=train_set)
def PavedDrive_encode1(x):
    if x == 'Y':
        return 1
    else:
        return 0

train_set['PavedDrive_yes'] = train_set.PavedDrive.apply(PavedDrive_encode1)

test_set['PavedDrive_yes'] = test_set.PavedDrive.apply(PavedDrive_encode1)

train_set.drop(['PavedDrive'],axis=1,inplace=True)
test_set.drop(['PavedDrive'],axis=1,inplace=True)

#PoolQC
train_set.drop(['PoolQC'],axis=1,inplace=True)
test_set.drop(['PoolQC'],axis=1,inplace=True)

#Fence
train_set.drop(['Fence'],axis=1,inplace=True)
test_set.drop(['Fence'],axis=1,inplace=True)

#MiscFeature
train_set.drop(['MiscFeature'],axis=1,inplace=True)
test_set.drop(['MiscFeature'],axis=1,inplace=True)

#SaleType
def SaleType_encode1(x):
    if x == 'WD' or x == 'CWD' or x == 'VWD':
        return 1
    else:
        return 0

def SaleType_encode2(x):
    if x == 'New':
        return 1
    else:
        return 0

train_set['Wd_SaleType'] = train_set.SaleType.apply(SaleType_encode1)
train_set['New_SaleType'] = train_set.SaleType.apply(SaleType_encode2)

test_set['Wd_SaleType'] = test_set.SaleType.apply(SaleType_encode1)
test_set['New_SaleType'] = test_set.SaleType.apply(SaleType_encode2)

train_set.drop(['SaleType'],axis=1,inplace=True)
test_set.drop(['SaleType'],axis=1,inplace=True)

#SaleCondition
def SaleCondition_encode1(x):
    if x == 'Normal':
        return 1
    else:
        return 0

def SaleCondition_encode2(x):
    if x == 'Partial':
        return 1
    else:
        return 0

train_set['Normal_SaleCondition'] = train_set.SaleCondition.apply(SaleCondition_encode1)
train_set['Partial_SaleCondition'] = train_set.SaleCondition.apply(SaleCondition_encode2)

test_set['Normal_SaleCondition'] = test_set.SaleCondition.apply(SaleCondition_encode1)
test_set['Partial_SaleCondition'] = test_set.SaleCondition.apply(SaleCondition_encode2)

train_set.drop(['SaleCondition'],axis=1,inplace=True)
test_set.drop(['SaleCondition'],axis=1,inplace=True)



# print(train_set.info())
# print(test_set.info())


"""
Treating all Null values

LotFrontage              258
GarageYrBlt               81
MasVnrArea                 8

For numeric features we can usually replace null values with a 0 but in some cases it is preferred to 
replace the null values with the mean of the column. Such as for GarageYrBlt.
"""
#LotFrontage
#print(train_set['LotFrontage'])
train_set['LotFrontage'].fillna(0,inplace=True)
test_set['LotFrontage'].fillna(0,inplace=True)


#GarageYrBlt

#print('\n',train_set.mean(axis=0)['GarageYrBlt'])
#sns.scatterplot('GarageYrBlt',target,data=train_set)
train_set['GarageYrBlt'].fillna(1978,inplace=True)
test_set['GarageYrBlt'].fillna(1978,inplace=True)

#MasVnrArea
#print(train_set['MasVnrArea'])
train_set['MasVnrArea'].fillna(0,inplace=True)
test_set['MasVnrArea'].fillna(0,inplace=True)


#nulls = pd.DataFrame(test_set.isnull())
#print(nulls.sum().sort_values(ascending=False))
#categoricals = train_set.select_dtypes(exclude=[np.number])
#print('\n',categoricals.count())

"""
Treat Remaining Null values in test_set.
BsmtHalfBath             2
BsmtFullBath             2
TotalBsmtSF              1
BsmtFinSF1               1
GarageCars               1
BsmtUnfSF                1
BsmtFinSF2               1
GarageArea               1
"""

#print(test_set['BsmtHalfBath'])
test_set['BsmtHalfBath'].fillna(0,inplace=True)
test_set['BsmtFullBath'].fillna(0,inplace=True)
test_set['TotalBsmtSF'].fillna(0,inplace=True)
test_set['BsmtFinSF1'].fillna(0,inplace=True)
test_set['GarageCars'].fillna(0,inplace=True)
test_set['BsmtUnfSF'].fillna(0,inplace=True)
test_set['BsmtFinSF2'].fillna(0,inplace=True)
test_set['GarageArea'].fillna(0,inplace=True)


#nulls = pd.DataFrame(test_set.isnull())
#print(nulls.sum().sort_values(ascending=False)[:7])
#categoricals = test_set.select_dtypes(exclude=[np.number])
#print('\n',categoricals.count())

plt.show()
