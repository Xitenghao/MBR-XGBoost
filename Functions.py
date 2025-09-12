#Load dependencies
from keras import Sequential
import numpy as np 
import skfda
import random
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree
from skfda.ml.regression import LinearRegression
from skfda.preprocessing.dim_reduction.variable_selection.maxima_hunting import (
    MaximaHunting,
    RelativeLocalMaximaSelector)
from skfda.representation.basis import BSplineBasis
from skfda.ml.regression import KNeighborsRegressor
from skfda.misc.kernels import uniform
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,)
from skfda.preprocessing.missing import MissingValuesInterpolation
from skfda.representation.basis import FDataBasis
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import matplotlib
import warnings
warnings.filterwarnings('ignore')

def data_deal(Yield):
    mto_df = pd.read_excel('...')
    mto_df = mto_df.iloc[:,4:]
    col_xy = list(mto_df.columns)
    col_x = list(mto_df.columns[0:15])
    mto_df['Yield'] = mto_df[Yield]
    mto_df['Yield_1'] = mto_df['Sethylene%']
    mto_df['ID'] = None
    k = 0
    for i, row in mto_df.iterrows():
        if pd.notna(mto_df.at[i, 'ID']):
            continue
        mto_df.at[i, 'ID'] = k
        in_row_x = row[col_x]
        for j in range(i+1, len(mto_df)):
            if (in_row_x == mto_df.loc[j, col_x]).all():
                if mto_df.at[j, 'TOS(min)'] > mto_df.at[j-1, 'TOS(min)']:
                    mto_df.at[j, 'ID'] = k
                else:
                    break
            else:
                break
        k += 1
        
    Info = (
        mto_df
        .groupby('ID')
        .agg(
            n=('TOS(min)', lambda x: sum(x.notna() & mto_df['Yield_1'].notna())),
            maxTOS=('TOS(min)', 'max'),
            minTOS=('TOS(min)', 'min'),
            # maxYield=('Yield', lambda x: x.max(skipna=True) if x.notna().any() else np.nan)
        )
        .reset_index()
    )
    
    grouped = mto_df.loc[:,['ZEOTYPE','TOS(min)','ID']].groupby('ID')
    max_tos = [grp['TOS(min)'].max() for i,grp in grouped]
    max_tos_ID = [grp.iloc[0,2] for i,grp in grouped]
    max_tos_z = [grp.iloc[0,0] for i,grp in grouped]

    max_zt_df = pd.DataFrame()
    max_zt_df['ID']=max_tos_ID
    max_zt_df['ZEOTYPE']=max_tos_z
    max_zt_df['max_tos']=max_tos
    
    use_ID = Info.loc[(Info['maxTOS'] <= 3000) & (Info['n'] >= 5) & (Info['maxTOS'] >= 300),'ID'].tolist()
    use_mto_df = mto_df[mto_df['ID'].isin(use_ID)]
    use_mto_df.reset_index(drop=True,inplace=True)
    
    Tmin = 0; Tmax = 300
    Grids = np.arange(Tmin, Tmax+1, 1)
    
    df_Curves = format_Curves(use_mto_df, use_ID, Grids)
    
    df_Curves = df_Curves.reset_index(drop=True)
    missing_count_per_row = df_Curves.isnull().sum(axis=1)
    threshold = len(Grids) * 0.5
    df_Curves = df_Curves[missing_count_per_row <= threshold]
    index_to_keep = df_Curves.index.tolist()

    df_Curves_copy = df_Curves.copy()
    df_diff = df_Curves_copy.diff(axis=1)
    threshold = 2 * df_diff.std().mean()
    outliers = np.abs(df_diff) > threshold
    outlier_indices = outliers.any(axis=1)

    df_Curves_copy[outliers] = np.nan
    df_Curves = df_Curves_copy
    
    data_tpts = Grids
    data_y = df_Curves.reset_index(drop=True)
    
    data_y[data_y < 0] = np.nan
    
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    
    nan_interp = MissingValuesInterpolation()
    data_y_f = nan_interp.fit_transform(data_y_f)
    
    fd_os = KernelSmoother(
        kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=6, kernel=uniform),
    ).fit_transform(data_y_f)
    data_y_kns = pd.DataFrame(fd_os.data_matrix.reshape(-1,len((fd_os.data_matrix[0]))),columns=data_tpts)
    
    data_y = data_y_kns
    
    X_name = ['Modification','AS','A/T','FDSi','Largest Ring sizes','MDa',
              'MDb','MDc','Mdi','CD','crystal size(μm)','reaction temp(°C)',
              'WHSV(h-1)']
    cla_name = ['Modification', 'AS', 'Largest Ring sizes', 'CD']
    num_name = [n for n in X_name if n not in cla_name]
    
    X_df = use_mto_df.drop_duplicates(subset='ID', keep='first')
    data_x = X_df[X_name].reset_index(drop=True)
    data_x = data_x.loc[index_to_keep]
    data_x = data_x.reset_index(drop=True)
    print('Data processing completed!')
    return data_tpts,data_x,data_y

'''Format the target variable y of the original data, including interpolation'''
def format_Curves(data_use, ID_use, Grids):
    # Initialize the Curves matrix
    nrow = len(ID_use)
    ncol = len(Grids)
    Curves = np.full((nrow, ncol), np.nan)
    for i, current_id in enumerate(ID_use):
        # Extract the index positions of the current ID curve
        idx_iID = data_use.index[data_use['ID'] == current_id].tolist()
        # Process the current ID curve
        numTOS = len(data_use.loc[idx_iID, 'TOS(min)'])
        TOS_iID = data_use.loc[idx_iID, 'TOS(min)'].values
        Yield_iID = data_use.loc[idx_iID, 'Yield'].values
        # Predicted time points
        Grids_pred = np.full(len(Grids), np.nan)
        tail_num = 1
        slope, _ = np.polyfit(TOS_iID[-tail_num:], Yield_iID[-tail_num:], 1)
        for iGrid, grid_point in enumerate(Grids):
            if grid_point > np.max(TOS_iID):
                break
            idx = np.sum(TOS_iID <= grid_point)
            if idx <= 1:
                if idx == 0:
                    val = Yield_iID[0]
                else:
                    val = Yield_iID[0] + (grid_point - TOS_iID[0]) * \
                          (Yield_iID[1] - Yield_iID[0]) / (TOS_iID[1] - TOS_iID[0])
            elif idx > numTOS-1:
                val = 0
            else:
                val = Yield_iID[idx-1] + (grid_point - TOS_iID[idx-1]) * \
                      (Yield_iID[idx] - Yield_iID[idx-1]) / (TOS_iID[idx] - TOS_iID[idx-1])
            Grids_pred[iGrid] = val
        # Store the results
        Curves[i, :] = Grids_pred
    # Set the row and column names of Curves
    index_names = ID_use
    column_names = ['T' + str(grid) for grid in Grids]
    df_Curves = pd.DataFrame(Curves, index=index_names, columns=column_names)
    # Replace -Inf and Inf with np.nan
    df_Curves.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_Curves

'''Define a function to split data'''
def split(n, split_rate, seed):
    if isinstance(split_rate, list) or isinstance(split_rate, np.ndarray):
        train_no = split_rate
        train_set = set(train_no)
        z_set = set(list(range(n)))
        test_no = list(z_set - train_set)
    elif split_rate > 0 and split_rate < 1:
        random.seed(seed)
        train_no = random.sample(range(0,n), round(n*split_rate))
        train_set = set(train_no)
        z_set = set(list(range(n)))
        test_no = list(z_set - train_set)
    return train_no, test_no

'''Define a function to standardize data'''
def rescale(x, scale_type):
    # Centering and scaling (subtract the mean and divide by the standard deviation)
    if scale_type == 0:
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    # Scale each column of x to the [0,1] interval
    elif scale_type == 1:
        # Compute min and max for each column
        min_vals = x.min(axis=0)
        max_vals = x.max(axis=0)
        # Compute range (max - min) for each column
        ranges = max_vals - min_vals
        # Avoid division by zero
        ranges[ranges == 0] = 1
        # Scale data to [0,1] range
        x = (x - min_vals) / ranges
    return x

'''Function to calculate R2, input data format is dataframe format'''
def R2(y_test, y_pred):
    numerator = np.nansum((y_test - y_pred)**2)
    denominator = np.nansum((y_test - np.nanmean(y_test))**2)
    try:
        R2 = 1 - (numerator / denominator)
    except:
        R2 = 0
    return R2

'''Implementation of linear functional regression and KNN methods'''
def fos_B(data_x, data_y, data_tpts, n_basis, order=4,
          test_size=0.2, seed=None, ite=None,
          model_type='l', n_neighbors=5):
    '''model_type: Available models include linear regression and KNN regression'''
    # Integrate functional data observations into a fixed format
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    # Create a spline function
    basis = BSplineBasis(
        n_basis=n_basis,
        order=n_basis)
    # Fit the functional data y to obtain basis coefficients
    basis_y = data_y_f.to_basis(basis)
    # Create an index array with the same number of rows as data_x
    indices = np.arange(data_x.shape[0])
    if test_size == 0:
        X_train = data_x
        y_train = basis_y
    # Use train_test_split to split data and indices
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data_x,
            basis_y,
            indices,  # Add indices as an additional parameter here
            test_size=test_size,
            random_state=seed[ite])
    # Create model
    if model_type == 'l':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == 'k':
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        X_train = X_train.to_numpy()
        if test_size != 0:
            X_test = X_test.to_numpy()
        model.fit(X_train, y_train)
    if test_size == 0:
        return {'basis_fun': basis,
                'basis_b': basis_y,
                'model': model}
    else:
        # Predict basis coefficients
        y_pred_b = model.predict(X_test)
        # Calculate predicted values from basis coefficients and basis functions
        y_pred = y_pred_b.to_grid(data_tpts).data_matrix
        # Reconstruct format
        y_pred = y_pred.reshape(-1, len((y_pred[0])))
        y_pred_df = pd.DataFrame(y_pred, columns=data_y.columns)
        y_test_obs = data_y.iloc[test_idx, :]
        # Calculate R-squared
        numerator = np.nansum((y_test_obs - y_pred_df) ** 2)
        denominator = np.nansum((y_test_obs - np.nanmean(y_test_obs)) ** 2)
        try:
            R2_score = 1 - (numerator / denominator)
        except:
            R2_score = 0
        # Calculate MISE
        mise_score = np.nanmean(np.nansum((y_test_obs - y_pred_df) ** 2, axis=1))

        mae_score = np.nanmean(np.nanmean(abs(y_test_obs - y_pred_df), axis=1))

        rmse_score = np.sqrt(np.nanmean(np.nanmean((y_test_obs - y_pred_df) ** 2, axis=1)))

        return {'basis_fun': basis,
                'basis_b': basis_y,
                'test_idx': test_idx,
                'model': model,
                'y_pred_b': y_pred_b,
                'y_pred': y_pred,
                'R2_score': R2_score,
                'mise_score': mise_score,
                'mae_score': mae_score,
                'rmse_score': rmse_score
                }

'''Multi output Decision Tree regression model'''
'''Loss function based on the loss of the base coefficients.'''
def DTR_B(data_x,data_y,data_tpts,n_basis,order=4,
          test_size=0.2,seed=None,ite=None,
          model_type='l',n_neighbors=5):
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    basis = BSplineBasis(
        n_basis=n_basis,
        order=order)
    basis_y = data_y_f.to_basis(basis)
    indices = np.arange(data_x.shape[0])
    if test_size==0:
        X_train = data_x
        y_train = basis_y
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data_x,
            basis_y,
            indices,  
            test_size=test_size,
            random_state=seed[ite])

    y_train_c = pd.DataFrame(y_train.coefficients,columns=list(range(n_basis)))
    model_dict = {}
    for i in range(n_basis):
        regr_i = DecisionTreeRegressor(max_depth=4)
        regr_i.fit(X_train, y_train_c.iloc[:,i])
        model_dict['regr_'+str(i)] = regr_i
    y_pred_c = []
    for key, model in model_dict.items():
        y_pred_c_i = list(model.predict(X_test))
        y_pred_c.append(y_pred_c_i)
    y_pred_c = np.array(y_pred_c).reshape(len(y_pred_c[0]),len(y_pred_c))
    basis = BSplineBasis(
        domain_range=(0.0, data_tpts[-1]),
        n_basis=n_basis,
        order=order)
    y_pred_b = FDataBasis(basis, y_pred_c)
    y_pred = y_pred_b.to_grid(data_tpts).data_matrix
    y_pred = y_pred.reshape(-1,len((y_pred[0])))
    y_pred_df = pd.DataFrame(y_pred,columns=data_y.columns)
    y_test_obs = data_y.iloc[test_idx,:]
    numerator = np.nansum((y_test_obs - y_pred_df)**2)
    denominator = np.nansum((y_test_obs - np.nanmean(y_test_obs))**2)
    try:
        R2_score = 1 - (numerator / denominator)
    except:
        R2_score = 0
    mise_score = np.nanmean(np.nansum((y_test_obs - y_pred_df)**2, axis=1))
    return {
            'test_idx':test_idx,
            'model_dict':model_dict,
            'y_pred_b':y_pred_b,
            'y_pred':y_pred,
            'R2_score':R2_score,
            'mise_score':mise_score}

'''Multi output RF model'''
'''Loss function based on the loss of the base coefficients.'''
def RSF_B(data_x,data_y,data_tpts,n_basis,order=4,
          test_size=0.2,seed=None,ite=None,
          n_estimators=100,max_depth = 4):
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    basis = BSplineBasis(
        n_basis=n_basis,
        order=order)
    basis_y = data_y_f.to_basis(basis)
    indices = np.arange(data_x.shape[0])
    if test_size==0:
        X_train = data_x
        y_train = basis_y
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data_x,
            basis_y,
            indices,
            test_size=test_size,
            random_state=seed[ite])

    y_train_c = pd.DataFrame(y_train.coefficients,columns=list(range(n_basis)))
    rsf_mult = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=n_estimators,
                              max_depth=max_depth,
                              random_state=seed[ite]))
    rsf_mult.fit(X_train, y_train_c)
    if test_size==0:
        return {'model':rsf_mult}
    else:
        y_pred_c= rsf_mult.predict(X_test)
        # y_pred_c = np.array(y_pred_c).reshape(len(y_pred_c[0]),len(y_pred_c))
        basis = BSplineBasis(
            domain_range=(0.0, data_tpts[-1]),
            n_basis=n_basis,
            order=order)
        y_pred_b = FDataBasis(basis, y_pred_c)
        y_pred = y_pred_b.to_grid(data_tpts).data_matrix
        y_pred = y_pred.reshape(-1,len((y_pred[0])))
        y_pred_df = pd.DataFrame(y_pred,columns=data_y.columns)
        y_test_obs = data_y.iloc[test_idx,:]
        numerator = np.nansum((y_test_obs - y_pred_df)**2)
        denominator = np.nansum((y_test_obs - np.nanmean(y_test_obs))**2)
        try:
            R2_score = 1 - (numerator / denominator)
        except:
            R2_score = 0
        mise_score = np.nanmean(np.nansum((y_test_obs - y_pred_df)**2, axis=1))
        return {
                'test_idx':test_idx,
                'model':rsf_mult,
                'y_pred_b':y_pred_b,
                'y_pred':y_pred,
                'R2_score':R2_score,
                'mise_score':mise_score}


'''Multi output GBDT model'''
'''Loss function based on the loss of the base coefficients.'''
def GBDT_B(data_x,data_y,data_tpts,n_basis,order=4,
          test_size=0.2,seed=None,ite=None,params=None):
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    basis = BSplineBasis(
        n_basis=n_basis,
        order=order)
    basis_y = data_y_f.to_basis(basis)
    indices = np.arange(data_x.shape[0])
    if test_size==0:
        X_train = data_x
        y_train = basis_y
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data_x,
            basis_y,
            indices,  
            test_size=test_size,
            random_state=seed[ite])

    y_train_c = pd.DataFrame(y_train.coefficients,columns=list(range(n_basis)))
    
    if params is None:
        gbdt_mult = MultiOutputRegressor(GradientBoostingRegressor(
                                                                   n_estimators=50,
                                                                   learning_rate=0.3,
                                                                   max_depth=6,
                                                                  ))
    else:
        gbdt_mult = MultiOutputRegressor(GradientBoostingRegressor(**params))
    gbdt_mult.fit(X_train, y_train_c)
    
    if test_size==0:
        return {'model':gbdt_mult}
    else:
        y_pred_c= gbdt_mult.predict(X_test)
        # y_pred_c = np.array(y_pred_c).reshape(len(y_pred_c[0]),len(y_pred_c))
        basis = BSplineBasis(
            domain_range=(0.0, data_tpts[-1]),
            n_basis=n_basis,
            order=order)
        y_pred_b = FDataBasis(basis, y_pred_c)
        y_pred = y_pred_b.to_grid(data_tpts).data_matrix
        y_pred = y_pred.reshape(-1,len((y_pred[0])))
        y_pred_df = pd.DataFrame(y_pred,columns=data_y.columns)
        y_test_obs = data_y.iloc[test_idx,:]
        numerator = np.nansum((y_test_obs - y_pred_df)**2)
        denominator = np.nansum((y_test_obs - np.nanmean(y_test_obs))**2)
        try:
            R2_score = 1 - (numerator / denominator)
        except:
            R2_score = 0
        mise_score = np.nanmean(np.nansum((y_test_obs - y_pred_df)**2, axis=1))
        mae_score = np.nanmean(np.nanmean(abs(y_test_obs - y_pred_df), axis=1))
        rmse_score = np.sqrt(np.nanmean(np.nanmean((y_test_obs - y_pred_df)**2, axis=1)))
        return {
                'test_idx':test_idx,
                'model':gbdt_mult,
                'y_pred_b':y_pred_b,
                'y_pred':y_pred,
                'R2_score':R2_score,
                'mise_score':mise_score,
                'mae_score':mae_score,
                'rmse_score':rmse_score}
    

'''Multi output XGB model'''
'''Loss function based on the loss of the base coefficients.'''
def XGB_B(data_x,data_y,data_tpts,n_basis,order=4,
          test_size=0.2,seed=None,ite=None,params=None):
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    basis = BSplineBasis(
        n_basis=n_basis,
        order=order)
    basis_y = data_y_f.to_basis(basis)
    indices = np.arange(data_x.shape[0])
    if test_size==0:
        X_train = data_x
        y_train = basis_y
    else:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data_x,
            basis_y,
            indices,
            test_size=test_size,
            random_state=seed[ite])

    y_train_c = pd.DataFrame(y_train.coefficients,columns=list(range(n_basis)))
    
    if params is None:
        params = {'learning_rate': 0.45,
                  'max_depth': 5,
                  'n_estimators': 100,
                  'min_child_weight': 3,
                  'seed': seed[ite]}
    xgb_mult = MultiOutputRegressor(
        xgb.XGBRegressor(objective='reg:squarederror',
                         **params))
    xgb_mult.fit(X_train, y_train_c)
    
    if test_size==0:
        return {'model':xgb_mult}
    else:
        y_pred_c= xgb_mult.predict(X_test)
        basis = BSplineBasis(
            domain_range=(0.0, data_tpts[-1]),
            n_basis=n_basis,
            order=order)
        y_pred_b = FDataBasis(basis, y_pred_c)
        y_pred = y_pred_b.to_grid(data_tpts).data_matrix
        y_pred = y_pred.reshape(-1,len((y_pred[0])))
        y_pred_df = pd.DataFrame(y_pred,columns=data_y.columns)
        y_test_obs = data_y.iloc[test_idx,:]
        numerator = np.nansum((y_test_obs - y_pred_df)**2)
        denominator = np.nansum((y_test_obs - np.nanmean(y_test_obs))**2)
        try:
            R2_score = 1 - (numerator / denominator)
        except:
            R2_score = 0
        mise_score = np.nanmean(np.nansum((y_test_obs - y_pred_df)**2, axis=1))
        mae_score = np.nanmean(np.nanmean(abs(y_test_obs - y_pred_df), axis=1))
        rmse_score = np.sqrt(np.nanmean(np.nanmean((y_test_obs - y_pred_df)**2, axis=1)))
        return {
                'test_idx':test_idx,
                'model':xgb_mult,
                'y_pred_b':y_pred_b,
                'y_pred':y_pred,
                'R2_score':R2_score,
                'mise_score':mise_score,
                'mae_score':mae_score,
                'rmse_score':rmse_score}


'''Define the MBR xgboost function loss as follows'''
# coding=utf-8
import xgboost as xgb
import numpy as np
# Coefficient of L2 in objective function.
global alpha
global data_tpts
global n_basis
global order
global delta

def global_init(tpts, n_b, od):
    """
    initialization of global variables.
    """
    global  data_tpts, n_basis, order
    data_tpts = tpts
    n_basis = n_b
    order = od

def hit_eval_loss(preds, dtrain_y, dtrain_c):
    """
    Computation of Objective Function
    preds: DataFrame object, where each row represents a set of basis coefficients.
    dtrain_y: DataFrame object, containing the observed values of y(t), with each row representing a curve.
    Requires data_tpts, n_basis, and order from outside the function.
    """
    preds = np.array(preds)
    c_train = dtrain_c
    c_hat = preds
    N, K = c_hat.shape
    
    basis = BSplineBasis(
        domain_range=(0.0, data_tpts[-1]),
        n_basis=n_basis,
        order=order)
    pred_b = FDataBasis(basis, c_hat)
    pred_y = pred_b.to_grid(data_tpts).data_matrix
    pred_y = pred_y.reshape(-1, len((pred_y[0])))
    pred_y_df = pd.DataFrame(pred_y, columns=dtrain_y.columns)
    
    L1 = np.nanmean(np.nansum((dtrain_y - pred_y_df)**2, axis=1))

    L2 = np.nansum(np.square(c_hat - c_train) / N)
    
    return {"loss_1": L1, "loss_2": L2}

def hit_grads(preds, dtrain_y, dtrain_c, loss_type=1):
    """
    Gradient Computation of custom objective function.
    For high speed running, the implementation utilizes built-in function 
    in `numpy` as much as possible (at the cost of readability).
    preds: DataFrame object, where each row represents a set of basis coefficients.
    dtrain_y: DataFrame object, containing the observed values of y(t), with each row representing a curve.
    dtrain_c: DataFrame object, containing the training data of basis coefficients, obtained by fitting the observed values of y(t) to B-spline basis functions.
    Requires data_tpts, n_basis, and order from outside the function.
    """
    chat = preds
    N, K = chat.shape

    L1_grad = np.zeros_like(chat)
    L1_hess = np.zeros_like(chat)
    basis = BSplineBasis(
        domain_range=(0.0, data_tpts[-1]),
        n_basis=n_basis,
        order=n_basis)
    basis_fv = basis(data_tpts)
    basis_fv = basis_fv.reshape(-1, len(basis_fv[0]))
    basis_fv = pd.DataFrame(basis_fv.T)
    if loss_type == 1:

        L1_grad = (-1 * (dtrain_y.values - (chat @ basis_fv.T).values) @ basis_fv.values)
        L1_hess = pd.DataFrame(np.zeros_like(chat), dtype='float32')
        for i_n in range(N):
            L1_hess.iloc[i_n,:] = ((2 * basis_fv.pow(2).sum(axis=0))).astype('float32')
        grad = L1_grad
        hess = L1_hess
    elif loss_type == 2:
        L2_grad = (-2 * (dtrain_c.values - chat.values))
        L2_hess = pd.DataFrame(2 * np.ones_like(chat.values))
        grad = L2_grad
        hess = L2_hess.to_numpy()
    else:
        print("Please set the correct loss function type,\n1:MISE,\n2:loss of c_hat,\n3:huber loss of y(t)_hat")
    grad = pd.DataFrame(grad.astype(np.float32))
    hess = pd.DataFrame(hess.astype(np.float32))
    return grad, hess
 
def hit_eval(model_list, dtest_list, y_test, dtest_c):
    pred_list = []
    for i, model in enumerate(model_list):
        pred = list(model.predict(dtest_list[i]))
        pred_list.append(pred)
    pred_list = np.array(pred_list)
    pred_list = pred_list.reshape(len(pred_list[0]), len(pred_list))
    lossv = hit_eval_loss(pred_list, y_test, dtest_c)['loss_1']
    return lossv

def print_eval(iters_num, test_idx, loss):
    print(f"# After {iters_num}th iteration:")
    print(f"\t\tloss: {loss}")
    print(f"test data:{test_idx}")

def MBR_XGB(data_x, data_y, model_params, 
        data_tpts, n_basis, order, 
        num_rounds=200, loss_type=1,
        test_size=0.2, seed=None,
        ite=None, silent=False):
    loss_result = {'loss': []}
    global_init(data_tpts, n_basis, order)
    
    # Integrate functional data observations into a fixed format
    data_y_f = skfda.FDataGrid(
        data_matrix=data_y,
        grid_points=data_tpts)
    # Create a spline function
    basis = BSplineBasis(
        n_basis=n_basis,
        order=order)
    # Fit the functional data y to obtain basis coefficients
    basis_y = data_y_f.to_basis(basis)
    
    # Create an index array with the same number of rows as data_x
    indices = np.arange(data_x.shape[0])
    if test_size == 0:
        X_train = data_x
        y_train = data_y
        c_train = basis_y
        dtrain_c = pd.DataFrame(c_train.coefficients)
        dtrain_y = y_train
    # Use train_test_split to split data and indices
    else:
        X_train, X_test, c_train, c_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data_x,
            basis_y,
            data_y,
            indices,  # Add indices as an additional parameter here
            test_size=test_size,
            random_state=seed[ite])
        dtrain_c = pd.DataFrame(c_train.coefficients)
        dtest_c = pd.DataFrame(c_test.coefficients)
        dtrain_y = y_train
    ## Initialize model
    model_list = []
    dtrain_list = []
    dtest_list = []
    for i in range(n_basis):
        # Integrate the basis coefficient training data into a DMatrix and add it to the DMatrix list
        dtrain = xgb.DMatrix(X_train, label=dtrain_c[i])
        dtrain_list.append(dtrain)
        if test_size != 0:
            dtest = xgb.DMatrix(X_test, label=dtest_c[i])
            dtest_list.append(dtest)
        # Build a model for each dataset (X, ck)
        model = xgb.Booster(model_params, [dtrain_list[i]])
        # model = xgb.train(model_params, dtrain, num_boost_round=0)
        model_list.append(model)
        
    for _ in range(num_rounds):
        # Loop through models to make predictions
        pred_total = pd.DataFrame()
        for i in range(n_basis):
            model = model_list[i]
            pred = model.predict(dtrain_list[i])
            pred_total[i] = pred
        # Calculate gradients and Hessian matrices based on model predictions and training data
        g, h = hit_grads(pred_total, dtrain_y, dtrain_c, loss_type=loss_type)
        # Loop to update and iterate

'''Compare the performance of various models'''
def compared(n, seed_z, data_x, data_y, 
             data_tpts, n_basis, order, 
             test_size, plot_flag=True,
             method_list = ['NNBR','lin_b','knn_b','rsf_b','xgb_b','MBR_XGB'],
             ensemble = False):
    random.seed(seed_z)
    seed = list(random.sample(list(range(1, 1001)), n))

    Index_all = {'method_name':[],
                 'R2_mean':[],
                 'R2_var':[],
                 'R2_max':[],
                 'R2_min':[],
                 'Mise_mean':[],
                 'Mise_var':[],
                 'Mae_mean':[],
                 'Mae_var':[],
                 'Rmse_mean':[],
                 'Rmse_var':[]}
    MISE_all = {}
    if 'NNBR' in method_list:
        R2_n = []
        mise_n = []
        y_pred_to_n = []
        test_idx_to_n = []
        for i in range(n):
            output = NNBR(data_x, data_y, data_tpts,
                          n_basis = n_basis, order = order, 
                          test_size = test_size, seed = seed, 
                          ite = i, hidden_nodes = [10,8,6], 
                          activations = ['sigmoid','sigmoid','sigmoid'], batch_size =16, 
                          epochs = 1000, early_stopping = False, 
                          early_patience = 10)
        mean_R2 = np.mean(R2_n)
        var_R2 = np.var(R2_n)
        max_R2 = np.max(R2_n)
        min_R2 = np.min(R2_n)
        mean_mise = np.mean(mise_n)
        var_mise = np.var(mise_n)
        

        Index_all['method_name'].append('fboost_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)
    
    if 'lin_b' in method_list:
        R2_l = []
        mise_l = []
        mae_l = []
        rmse_l = []
        y_pred_to_l = []
        test_idx_to_l = []
        for i in range(n):
            out_put_l = fos_B(data_x,data_y,data_tpts,
                            n_basis=n_basis,order=order,
                            test_size=test_size,seed=seed,ite=i,
                            model_type='l')
            R2_l.append(out_put_l['R2_score'])
            mise_l.append(out_put_l['mise_score'])
            y_pred_to_l.append(out_put_l['y_pred'])
            test_idx_to_l.append(out_put_l['test_idx'])
            mae_l.append(out_put_l['mae_score'])
            rmse_l.append(out_put_l['rmse_score'])
        mean_R2 = np.mean(R2_l)
        var_R2 = np.var(R2_l)
        max_R2 = np.max(R2_l)
        min_R2 = np.min(R2_l)
        mean_mise = np.mean(mise_l)
        var_mise = np.var(mise_l)
        mean_mae = np.mean(mae_l)
        var_mae = np.var(mae_l)
        mean_rmse = np.mean(rmse_l)
        var_rmse = np.var(rmse_l)
        
        MISE_all['LIN'] = mise_l
        
        Index_all['method_name'].append('lin_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)
        Index_all['Mae_mean'].append(mean_mae)
        Index_all['Mae_var'].append(var_mae)
        Index_all['Rmse_mean'].append(mean_rmse)
        Index_all['Rmse_var'].append(var_rmse)
    
    if 'knn_b' in method_list:
        R2_k = []
        mise_k = []
        mae_k = []
        rmse_k = []
        y_pred_to_k = []
        test_idx_to_k = []
        for i in range(n):
            out_put_k = fos_B(data_x,data_y,data_tpts,
                            n_basis=n_basis,order=n_basis,
                            test_size=test_size,seed=seed,ite=i,
                            model_type='k', n_neighbors=5)
            R2_k.append(out_put_k['R2_score'])
            mise_k.append(out_put_k['mise_score'])
            y_pred_to_k.append(out_put_k['y_pred'])
            test_idx_to_k.append(out_put_k['test_idx'])
            mae_k.append(out_put_k['mae_score'])
            rmse_k.append(out_put_k['rmse_score'])
        mean_R2 = np.mean(R2_k)
        var_R2 = np.var(R2_k)
        max_R2 = np.max(R2_k)
        min_R2 = np.min(R2_k)
        mean_mise = np.mean(mise_k)
        var_mise = np.var(mise_k)
        mean_mae = np.mean(mae_k)
        var_mae = np.var(mae_k)
        mean_rmse = np.mean(rmse_k)
        var_rmse = np.var(rmse_k)

        MISE_all['KNN'] = mise_k
        
        Index_all['method_name'].append('knn_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)
        Index_all['Mae_mean'].append(mean_mae)
        Index_all['Mae_var'].append(var_mae)
        Index_all['Rmse_mean'].append(mean_rmse)
        Index_all['Rmse_var'].append(var_rmse)
        
    if 'dtr_b' in method_list:
        R2_d = []
        mise_d = []
        y_pred_to_d = []
        test_idx_to_d = []
        for i in range(n):
            out_put_d = DTR_B(data_x,data_y,data_tpts,
                            n_basis=5,order=4,
                            test_size=test_size,seed=seed,ite=i)
            R2_d.append(out_put_d['R2_score'])
            mise_d.append(out_put_d['mise_score'])
            y_pred_to_d.append(out_put_d['y_pred'])
            test_idx_to_d.append(out_put_d['test_idx'])
        mean_R2 = np.mean(R2_d)
        var_R2 = np.var(R2_d)
        max_R2 = np.max(R2_d)
        min_R2 = np.min(R2_d)
        mean_mise = np.mean(mise_d)
        var_mise = np.var(mise_d)

        Index_all['method_name'].append('dtr_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)

    if 'rsf_b' in method_list:
        R2_r = []
        mise_r = []
        y_pred_to_r = []
        test_idx_to_r = []
        for i in range(n):
            out_put_r = RSF_B(data_x,data_y,data_tpts,
                            n_basis=n_basis,order=order,
                            test_size=test_size,seed=seed,ite=i,
                            n_estimators=100,max_depth = 6)
            R2_r.append(out_put_r['R2_score'])
            mise_r.append(out_put_r['mise_score'])
            y_pred_to_r.append(out_put_r['y_pred'])
            test_idx_to_r.append(out_put_r['test_idx'])
        mean_R2 = np.mean(R2_r)
        var_R2 = np.var(R2_r)
        max_R2 = np.max(R2_r)
        min_R2 = np.min(R2_r)
        mean_mise = np.mean(mise_r)
        var_mise = np.var(mise_r)

        Index_all['method_name'].append('rsf_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)

    if 'gbdt_b' in method_list:
        R2_g = []
        mise_g = []
        mae_g = []
        rmse_g = []
        y_pred_to_g = []
        test_idx_to_g = []
        for i in range(n):
            out_put_g = GBDT_B(data_x,data_y,data_tpts,
                            n_basis=n_basis,order=order,
                            test_size=test_size,seed=seed,ite=i)
            R2_g.append(out_put_g['R2_score'])
            mise_g.append(out_put_g['mise_score'])
            y_pred_to_g.append(out_put_g['y_pred'])
            test_idx_to_g.append(out_put_g['test_idx'])
            mae_g.append(out_put_g['mae_score'])
            rmse_g.append(out_put_g['rmse_score'])
        mean_R2 = np.mean(R2_g)
        var_R2 = np.var(R2_g)
        max_R2 = np.max(R2_g)
        min_R2 = np.min(R2_g)
        mean_mise = np.mean(mise_g)
        var_mise = np.var(mise_g)
        mean_mae = np.mean(mae_g)
        var_mae = np.var(mae_g)
        mean_rmse = np.mean(rmse_g)
        var_rmse = np.var(rmse_g)

        MISE_all['GBDT'] = mise_g
        
        Index_all['method_name'].append('gbdt_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)
        Index_all['Mae_mean'].append(mean_mae)
        Index_all['Mae_var'].append(var_mae)
        Index_all['Rmse_mean'].append(mean_rmse)
        Index_all['Rmse_var'].append(var_rmse)
        
    if 'xgb_b' in method_list:
        R2_x = []
        mise_x = []
        mae_x = []
        rmse_x = []
        y_pred_to_x = []
        test_idx_to_x = []
        for i in range(n):
            out_put_x = XGB_B(data_x,data_y,data_tpts,
                            n_basis=n_basis,order=order,
                            test_size=test_size,seed=seed,ite=i)
            R2_x.append(out_put_x['R2_score'])
            mise_x.append(out_put_x['mise_score'])
            y_pred_to_x.append(out_put_x['y_pred'])
            test_idx_to_x.append(out_put_x['test_idx'])
            mae_x.append(out_put_x['mae_score'])
            rmse_x.append(out_put_x['rmse_score'])
        mean_R2 = np.mean(R2_x)
        var_R2 = np.var(R2_x)
        max_R2 = np.max(R2_x)
        min_R2 = np.min(R2_x)
        mean_mise = np.mean(mise_x)
        var_mise = np.var(mise_x)
        mean_mae = np.mean(mae_x)
        var_mae = np.var(mae_x)
        mean_rmse = np.mean(rmse_x)
        var_rmse = np.var(rmse_x)

        MISE_all['XGB'] = mise_x
        
        Index_all['method_name'].append('xgb_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)
        Index_all['Mae_mean'].append(mean_mae)
        Index_all['Mae_var'].append(var_mae)
        Index_all['Rmse_mean'].append(mean_rmse)
        Index_all['Rmse_var'].append(var_rmse)
        
    if 'MBR_XGB' in method_list:
        model_params = {'eta': 0.15,
                'max_depth': 5,
                "min_child_weight": 5,
                'objective': 'reg:squarederror'}
        R2_mx = []
        mise_mx = []
        mae_mx = []
        rmse_mx = []
        y_pred_to_mx = []
        test_idx_to_mx = []
        for i in range(n):
            out_put_mx = MBR_XGB(data_x, data_y, model_params, 
                                data_tpts, n_basis=n_basis, order=order, 
                                num_rounds=200,loss_type=1,
                                test_size=0.2,seed=seed,
                                ite=i,silent=True)
            R2_mx.append(out_put_mx['R2_score'])
            mise_mx.append(out_put_mx['mise_score'])
            y_pred_to_mx.append(out_put_mx['y_pred'])
            test_idx_to_mx.append(out_put_mx['test_idx'])
            mae_mx.append(out_put_mx['mae_score'])
            rmse_mx.append(out_put_mx['rmse_score'])
        mean_R2 = np.mean(R2_mx)
        var_R2 = np.var(R2_mx)
        max_R2 = np.max(R2_mx)
        min_R2 = np.min(R2_mx)
        mean_mise = np.mean(mise_mx)
        var_mise = np.var(mise_mx)
        mean_mae = np.mean(mae_mx)
        var_mae = np.var(mae_mx)
        mean_rmse = np.mean(rmse_mx)
        var_rmse = np.var(rmse_mx)
        
        MISE_all['MBR-XGB'] = mise_mx
        
        Index_all['method_name'].append('fboost_b')
        Index_all['R2_mean'].append(mean_R2)
        Index_all['R2_var'].append(var_R2)
        Index_all['R2_max'].append(max_R2)
        Index_all['R2_min'].append(min_R2)
        Index_all['Mise_mean'].append(mean_mise)
        Index_all['Mise_var'].append(var_mise)
        Index_all['Mae_mean'].append(mean_mae)
        Index_all['Mae_var'].append(var_mae)
        Index_all['Rmse_mean'].append(mean_rmse)
        Index_all['Rmse_var'].append(var_rmse)

    if plot_flag:
        save_folder = 'image'
        matplotlib.rcParams['font.size'] = 22
        import os
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder_1):
            os.makedirs(save_folder_1)
        for k in range(n):
            m = k
            y_pred_l = y_pred_to_l[m]
            y_pred_k = y_pred_to_k[m]
            y_pred_g = y_pred_to_g[m]
            y_pred_x = y_pred_to_x[m]
            y_pred_mx = y_pred_to_mx[m]
            test_idx = test_idx_to_mx[m]

            for i, t_idx in enumerate(test_idx):
                curves_n = data_y.iloc[t_idx, :].tolist()
                curves_pred_l = y_pred_l[i].tolist()
                curves_pred_k = y_pred_k[i].tolist()
                curves_pred_g = y_pred_g[i].tolist()
                curves_pred_x = y_pred_x[i].tolist()
                curves_pred_mx = y_pred_mx.iloc[i,:].tolist()

                fig, ax = plt.subplots(figsize=(10, 8),dpi=300)
                ax.plot(range(len(curves_n)), curves_n, ls='-', c='black', label='Actual', linewidth=3)
                ax.plot(range(len(curves_pred_l)), curves_pred_l, ls='-', c='#EAB67A', label='LIN model', linewidth=2)
                ax.plot(range(len(curves_pred_g)), curves_pred_g, ls='--', c='#B38EBB', label='GBDT model', linewidth=2)
                ax.plot(range(len(curves_pred_k)), curves_pred_k, ls=':', c='#6E8FB2', label='KNN model', linewidth=2)
                ax.plot(range(len(curves_pred_x)), curves_pred_x, ls='-.', c='#3CA222', label='XGB model', linewidth=2)
                ax.plot(range(len(curves_pred_mx)), curves_pred_mx, ls='-', c='#EF4968', label='MBR-XGB model', linewidth=3)
                ax.set_xlim(0, len(curves_n))
                ax.set_xlabel('TOS (min)',fontsize=26)
                ax.set_ylabel('Sethylene selectivity',fontsize=26)
                ax.set_xticks([5,100,200,300])
                
                y_min = min(min(curves_n), min(curves_pred_l),min(curves_pred_k), min(curves_pred_g), min(curves_pred_mx), )*0.5
                y_max = max(max(curves_n), max(curves_pred_l),max(curves_pred_k), max(curves_pred_g), max(curves_pred_mx))*1.35
                ax.set_ylim(y_min, y_max)
                if t_idx==0:
                    ax.legend(loc='upper right', fontsize=19)
                plt.tight_layout()
                r2_mx = R2_mx[m]
                plt.savefig(os.path.join(save_folder, f'curve_{t_idx+1}_{k}.png'))
                plt.close()  
        return MISE_all,Index_all