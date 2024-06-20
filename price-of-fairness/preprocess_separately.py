#import scipy.io
import socket
import pandas as pd
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import LabelEncoder, normalize, normalize
#from numpy.linalg import inv, pinv, norm, svd, qr


# hide warnings
pd.options.mode.chained_assignment = None  # default='warn'

# global variables
VERBOSE = False

###############################################################################
## Preprocessing and misselaneous functions

if socket.gethostname() == "localhost":
    DATASET_PATH = "../datasets"
else:
    DATASET_PATH = "/scratch/fair-CSS-datasets"
#end if

def remove_string_columns(df):
    non_string_columns = df = select_dtypes(exclude=['object'])
    return non_string_columns
#end remove_string_columns()

def one_hot_encoder(df):
    return pd.get_dummies(df)
#end one_hot_encode()

def column_contains_string(df, column_name):
    return df[column_name].apply(lambda x: isinstance(x, str)).any()
#end column_contains_string()

def convert_to_datetimes_and_calc_days(df, column_names):
    base_date = pd.to_datetime('1970-01-01 00:00:00',  format='%Y-%m-%d %H:%M:%S')

    for col in column_names:
        # Ensure the column is in datetime format and handle missing values
        df[col] = pd.to_datetime(df[col],  format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Calculate the difference in days since 1900-01-01
        df[col] = (df[col] - base_date).dt.days

        # Replace NaT with NaN
        df[col].replace({pd.NaT: np.nan}, inplace=True)

    return df
#end convert_to_datetimes_and_calc_days()

def convert_to_dates_and_calc_days(df, column_names):
    base_date = pd.to_datetime('1900-01-01')

    for col in column_names:
        # Ensure the column is in datetime format and handle missing values
        df[col] = pd.to_datetime(df[col], errors='coerce')

        # Calculate the difference in days since 1900-01-01
        df[col] = (df[col] - base_date).dt.days

        # Replace NaT with NaN
        df[col].replace({pd.NaT: np.nan}, inplace=True)

    return df
#end convert_to_dates_and_calc_days()

###############################################################################
def preprocess_heart_dataset():
    df = pd.read_csv(DATASET_PATH+"/heart-disease/cleveland.data")

    if VERBOSE:
        print("--------------------------")
        print("Before pre-processing data")
        #instances and attributes
        print('dataset stats', df.shape)
        #sex statistics
        print(df.groupby(['sex'], sort=False).size())
        # missing values
        print('missing values',Counter(df[df.eq('?')].any(axis=1)))
        print('Column types:', df.dtypes)
    #end if

    #delete all rows with missing value
    df_new = df.replace('?', np.nan).dropna()
    df_new.reset_index(drop=True, inplace=True)

    #encoding labels to integer values
    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    df_new = one_hot_encoder(df_new)


    if VERBOSE:
        print("--------------------------")
        print("After pre-processing data")
        #instances and attributes
        print('dataset stats', df_new.shape)
        #sex statistics
        print(df_new.groupby(['sex'], sort=False).size())
        print("--------------------------\n\n")
    #end if
    
    grouped = df_new.groupby('sex', sort=False)

    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)
    M_df = df_new.drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    M_values = M_df.to_numpy()
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = normalize(M_values, axis=0, norm='l2')
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    M_values = M_values.round(decimals=5)
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return M_values,A_values,B_values
#end proprocess_heart_dataset()

def preprocess_adult_dataset():
    df = pd.read_csv(DATASET_PATH+"/adult/adult.data")

    if VERBOSE:
        print("--------------------")
        print("dataset: Adult census -- before preprocessing")
        print(df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    #encoding labels to integer values
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df = one_hot_encoder(df)

    #normalize data
    #df_norm = unit_norm(df, 'sex')
    #rounding decimals
    #df_norm = df_norm.round(decimals=5)

    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df.shape)
        #print(df_norm.groupby(['sex'], sort=False).size())
        print("--------------------------\n\n")
    #end if
    
    grouped = df.groupby('sex', sort=False)

    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    M_values = df.to_numpy()
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    M_values = normalize(M_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    return M_values, A_values, B_values
#end proprocess_adult_dataset()

def preprocess_autism_dataset():
    df = pd.read_csv(DATASET_PATH+"/autism/autism.csv")

    #print stats before preprocessing
    if VERBOSE:
        print("--------------------")
        print("dataset: Autism -- before preprocessing")
        print(df.shape)
        print(df.groupby(['gender'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    df_new = df.copy()
    df_new = df_new.replace('?', np.nan).dropna()

    #encoding labels to integer values
    le = LabelEncoder()
    df_new['gender'] = le.fit_transform(df_new['gender'])
    df_new = one_hot_encoder(df_new)

    #print stats after preprocessing
    if VERBOSE:
        print("--------------------")
        print("dataset: Autism -- after preprocessing")
        print(df_new.shape)
        print(df_new.groupby(['gender'], sort=False).size())
        print("--------------------------\n\n")
    #end if

    grouped = df_new.groupby('gender', sort=False)
    A_df = grouped.get_group(1.0).drop('gender', axis=1)
    B_df = grouped.get_group(0.0).drop('gender', axis=1)
    M_df = df_new.drop('gender', axis=1)

    #convert remaining columns to numpy arrays
    M_values = M_df.to_numpy()
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    A_values = normalize(A_values, axis=0, norm='l2').round(decimals=5)
    B_values = normalize(B_values, axis=0, norm='l2').round(decimals=5)
    M_values = normalize(M_values, axis=0, norm='l2').round(decimals=5)

    return M_values, A_values, B_values
#end preprocess_autism_dataset()


def preprocess_german_credit_dataset():
    df = pd.read_csv(DATASET_PATH+"/german-credit/german.data", delimiter=" ", header=None)

    #Adding column names
    df.columns = ['checkingAcc', 'durMonth', 'creditHist', 'purpose', 'creditAmt',\
                  'savingsAcc', 'employmentSince', 'instalRate', 'personalStatus',\
                  'otherDebtors', 'recidenceSize', 'property', 'age', 'instalPlans',\
                  'housing',\
                  'numOfCreditCards', 'job', 'dependents', 'telephone', 'foreignWorker',\
                  'decision']

    #adding an empty column
    df['sex'] = ''

    rows, cols = df.shape
    for r in range(rows):
        if (df['personalStatus'][r] == 'A92' or df['personalStatus'][r] == 'A95'):
            df['sex'][r] = 'female'
        else:
            df['sex'][r] = 'male'
        #end if
    #end for

    if VERBOSE:
        print("--------------------")
        print("dataset: German credit -- before preprocessing")
        print(df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:',df.dtypes)
    #end if
   
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df = one_hot_encoder(df)

    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df.shape)
        #print(df_norm.groupby(['sex'], sort=False).size())
        print("--------------------------\n\n")
    #endif

    #make a copy of dataframe
    df_new = df.copy()

    # drop protexted attributes
    grouped = df_new.groupby('sex', sort=False)
    A_df = grouped.get_group(1).drop('sex', axis=1)
    B_df = grouped.get_group(0).drop('sex', axis=1)
    M_df = df_new.drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    M_values = M_df.to_numpy()
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
   
    M_values = normalize(M_values, axis=0, norm='l2')
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    M_values = M_values.round(decimals=5)
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return M_values, A_values,B_values
#end process_german_credit_dataset()


def preprocess_credit_card_dataset():
    df = pd.read_csv(DATASET_PATH+"/credit-card/credit-card.csv")
    df.rename(columns = {'SEX':'sex'}, inplace = True)

    if VERBOSE:
        print("----------------------------")
        print("Credit card dataset")
        print("instances and attributes:",df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df = one_hot_encoder(df)

    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df.shape)
        #print(df_norm.groupby(['sex'], sort=False).size())
        print("--------------------------\n\n")
    #end if
    
    grouped = df.groupby('sex', sort=False)
    A_df = grouped.get_group(1).drop('sex', axis=1)
    B_df = grouped.get_group(0).drop('sex', axis=1)
    M_df = df.drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = M_df.to_numpy()
    
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    M_values = normalize(M_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    return M_values, A_values, B_values

def preprocess_student_entrance_dataset():
    df = pd.read_csv(DATASET_PATH+"/student-entrance/student_entrance.csv", 
                     delimiter=",")
    if VERBOSE:
        print("----------------------------")
        print("Student entrance dataset")
        print(df.shape)
        print(df.groupby(['Gender'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    # remove NaN values
    df_new = df.copy()
    df_new = df.replace('?', np.NaN).dropna()

    #assign strings a numeric value
    le = LabelEncoder()
    #for col in df_new.columns:
    #    df_new[col] = le.fit_transform(df_new[col])
    #end for
    df_new['Gender'] = le.fit_transform(df_new['Gender'])
    df_new = one_hot_encoder(df_new)

    if VERBOSE:
        print("---------------------")
        print("After preprocessing data")
        print(df_new.shape)
        print(df_new.groupby(['Gender'], sort=False).size())
        print("--------------------------\n\n")
    #end if

    #write to a csv file
    grouped = df_new.groupby('Gender', sort=False)
    A_df = grouped.get_group(1.0).drop('Gender', axis=1)
    B_df = grouped.get_group(0.0).drop('Gender', axis=1)
    M_df = df_new.drop('Gender', axis=1)

    # Convert remaining columns to numpy arrays
    M_values = M_df.to_numpy()
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
   
    M_values = normalize(M_values, axis=0, norm='l2')
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    M_values = M_values.round(decimals=5)
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)

    return M_values, A_values, B_values
#end preprocess_student_entrance_dataset()

def preprocess_student_perf_dataset():
    df = pd.read_csv(DATASET_PATH+"/student/student-por.csv", delimiter=";")

    # remove duplicate columns
    df_new = df.copy()
    #df_new = df_new.drop(columns=['first', 'last'])

    if VERBOSE:
        print("----------------------------")
        print("Student performance dataset")
        print(df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    #assign strings a numeric value
    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    df_new = one_hot_encoder(df_new)
        
    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df_new.shape)
        #print(df_norm.groupby(['sex'], sort=False).size())
        print("--------------------------\n\n")
    #end if

    grouped = df_new.groupby('sex', sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)
    M_df = df_new.drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    M_values = M_df.to_numpy()
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
   
    M_values = normalize(M_values, axis=0, norm='l2')
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    M_values = M_values.round(decimals=5)
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return M_values, A_values,B_values
#end preprocess_student_perf_dataset()

def preprocess_compas_dataset():
    df = pd.read_csv(DATASET_PATH+"/compas/compas-scores.csv")

    if VERBOSE:
        print("-----------------------")
        print("Compas recividism")
        print(df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    date_columns=['compas_screening_date', 'dob', 'c_jail_in', 'c_jail_out', 
                  'c_offense_date', 'vr_offense_date', 'v_screening_date', 
                  'screening_date']
    df=convert_to_dates_and_calc_days(df, date_columns)

    df_new = df[date_columns+['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 
                              'decile_score', 'juv_misd_count',
                              'juv_other_count', 'priors_count',
                              'days_b_screening_arrest', 'c_days_from_compas',
                              'c_charge_degree', 'c_charge_desc', 'is_recid',
                              'r_charge_degree', 'is_violent_recid',
                              'v_type_of_assessment', 'v_decile_score',
                              'v_score_text', 'type_of_assessment',
                              'decile_score.1', 'score_text']]
    df_new  = df_new.replace('?', np.nan).dropna()
    df_new.reset_index(drop=True, inplace=True)

    #assign strings a numeric value
    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    df_new = one_hot_encoder(df_new)
    contanins_nan = df_new.isnull().values.any()
    if VERBOSE:
        print(contanins_nan)
    #end if

    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df_new.shape)
        #print(df_norm.groupby(['sex'], sort=False).size())
        print("--------------------------\n\n")
    #end if

    grouped = df_new.groupby('sex', sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)
    M_df = df_new.drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = M_df.to_numpy()

    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    M_values = normalize(M_values, axis=0, norm='l2')
    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    return M_values, A_values, B_values
#end preprocess_compas_dataset()

def preprocess_communities_dataset():
    df = pd.read_csv(DATASET_PATH+"/communities/communities.data")

    if VERBOSE:
        print("-----------------------")
        print("Communities and crime")
        print(df.shape)
        print('Column types:', df.dtypes)
    #end if

    # replace unknown values with zero
    df_new=df.drop(columns=['county', 'community', 'communityname',
                            'LemasSwornFT', 'LemasSwFTPerPop',
                            'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop',
                            'LemasTotalReq', 'LemasTotReqPerPop',
                            'PolicReqPerOffic', 'PolicPerPop',
                            'RacialMatchCommPol', 'PctPolicWhite',
                            'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian',
                            'PctPolicMinor', 'OfficAssgnDrugUnits',
                            'NumKindsDrugsSeiz', 'PolicAveOTWorked',
                            'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
                            'LemasGangUnitDeploy', 'PolicBudgPerPop'])
    df_new = df.replace('?', 0)

    # introduce two new columns to identify majority white and majority
    # black neighbourhoods
    #df_new['majorityWhite'] = ''
    white_or_else=[]

    rows, cols = df_new.shape
    for r in range(rows):
        if df_new['racePctWhite'][r] >= 0.5:
            white_or_else.append(1.0)
        else:
            white_or_else.append(0.0)
        #end if
    #end for
    df_new['majorityWhite'] = white_or_else


    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df_new.shape)
        print(df_new.groupby(['majorityWhite'], sort=False).size())
        print("--------------------------\n\n")
    #end if
    
    grouped = df_new.groupby(['majorityWhite'], sort=False)
    A_df = grouped.get_group(1.0).drop(columns=['majorityWhite', 'racePctWhite',
                                                'racepctblack' ], axis=1)
    B_df = grouped.get_group(0.0).drop(columns=['majorityWhite', 'racePctWhite',
                                                'racepctblack' ], axis=1)
    M_df = df_new.drop(columns=['majorityWhite', 'racePctWhite',
                                'racepctblack'], axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = M_df.to_numpy()

    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    M_values = normalize(M_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    
    return M_values, A_values, B_values


    
def preprocess_recidivism_juvenile_dataset():
    df = pd.read_csv(DATASET_PATH+"/recividism-juvenile/recividism-juvenile.csv")

    if VERBOSE:
        print("-----------------------")
        print("Recividism juvenile")
        print(df.shape)
        print(df.groupby(['V1_sexe'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    date_columns=['V10_data_naixement', 'V22_data_fet',
                  'V30_data_inici_programa','V31_data_fi_programa'] 
    #, 'V55_SAVRYdata', 'V117_rein_data_fet_2013', 'V101_rein_data_fet_2015']
    df=convert_to_datetimes_and_calc_days(df, date_columns)

    df_new=df[['V1_sexe','V2_estranger','V3_nacionalitat',
               'V5_edat_fet_agrupat', 'V6_provincia', 'V7_comarca',
               'V8_edat_fet','V9_edat_final_programa','V10_data_naixement',
               'V11_antecedents', 'V12_nombre_ante_agrupat',
               'V13_nombre_fets_agrupat','V14_fet',
               'V15_fet_agrupat','V16_fet_violencia', 'V17_fet_tipus',
               'V19_fets_desagrupats', 'V20_nombre_antecedents',
               'V21_fet_nombre','V22_data_fet', 'V23_territori', 'V24_programa',
               'V25_programa_mesura', 'V27_durada_programa_agrupat',
               'V28_temps_inici', 'V29_durada_programa',
               'V30_data_inici_programa', 'V31_data_fi_programa']]
    df_new = df_new.replace('?', np.nan).dropna()
    df_new.reset_index(drop=True, inplace=True)

    le = LabelEncoder()
    df_new['V1_sexe'] = le.fit_transform(df_new['V1_sexe'])
    df_new=one_hot_encoder(df_new)

    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df_new.shape)
        print(df_new.groupby(['V1_sexe'], sort=False).size())
    #end if

    grouped = df_new.groupby(['V1_sexe'], sort=False)
    A_df = grouped.get_group(1.0).drop('V1_sexe', axis=1)
    B_df = grouped.get_group(0.0).drop('V1_sexe', axis=1)
    M_df = df_new.drop('V1_sexe', axis=1)
    
    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = M_df.to_numpy()
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    M_values = normalize(M_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    
    return M_values, A_values, B_values

def preprocess_meps_h181_dataset():
    df = pd.read_csv(DATASET_PATH+"/meps/h181.csv")
    df.rename(columns = {'SEX':'sex'}, inplace = True)

    if VERBOSE:
        print("-----------------------")
        print("MEPS - h181 ")
        print(df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    df_new = df.copy()

    #assign strings a numeric value
    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])

    # print stats after preprocessing
    if VERBOSE: 
        print("---------------------")
        print("After normalizing data")
        print(df_new.shape)
        print(df_new.groupby(['sex'], sort=False).size())
    #end if

    grouped = df_new.groupby(['sex'], sort=False)
    A_df = grouped.get_group(2).drop('sex', axis=1)
    B_df = grouped.get_group(1).drop('sex', axis=1)
    M_df = df_new.drop('sex', axis=1)
    
    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = M_df.to_numpy()
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    M_values = normalize(M_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    
    return M_values, A_values, B_values
#end preprocess_meps_h181_dataset() 
                     
def preprocess_meps_h192_dataset():
    df = pd.read_csv(DATASET_PATH+"/fair-CSS-datasets/meps/h192.csv")
    df.rename(columns = {'SEX':'sex'}, inplace = True)

    if VERBOSE:
        print("-----------------------")
        print("MEPS - h192 ")
        print(df.shape)
        print(df.groupby(['sex'], sort=False).size())
        print('Column types:', df.dtypes)
    #end if

    df_new = df.copy()

    #assign strings a numeric value
    le = LabelEncoder()
    for col in df_new.columns:
        if df_new[col].dtype == object:
            df_new[col] = le.fit_transform(df_new[col])
    #end for

    #round to 5 decimals
    df_norm =  df_norm.round(decimals=5)

    # print stats after preprocessing
    if VERBOSE:
        print("---------------------")
        print("After normalizing data")
        print(df_norm.shape)
        print(df_norm.groupby(['sex'], sort=False).size())
        print("---------------------\n\n")
    #end if

    grouped = df_norm.groupby(['sex'], sort=False)
    A_df = grouped.get_group(2).drop('sex', axis=1)
    B_df = grouped.get_group(1).drop('sex', axis=1)
    M_df = df_norm.drop('sex', axis=1)
    
    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    M_values = M_df.to_numpy()

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    M_values = M_values.round(decimals=5)
    
    return M_values, A_values, B_values
#end preprocess_meps_h192_dataset()

###############################################################################
def get_dataset(dataset):
    if 'heart' in dataset:
        return preprocess_heart_dataset()
    elif dataset == 'adult':
        return preprocess_adult_dataset()
    elif dataset == 'autism':
        return preprocess_autism_dataset()
    elif dataset == 'german-credit':
        return preprocess_german_credit_dataset()
    elif dataset == 'credit-card':
        return preprocess_credit_card_dataset()
    elif dataset == 'student-entrance':
        return preprocess_student_entrance_dataset()
    elif dataset == 'student-perf':
        return preprocess_student_perf_dataset()
    elif dataset == 'compas':
        return preprocess_compas_dataset()
    elif dataset == 'communities':
        return preprocess_communities_dataset()
    elif dataset == 'recidivism-juvenile':
        return preprocess_recidivism_juvenile_dataset()
    elif dataset == 'meps-h181':
        return preprocess_meps_h181_dataset()
    elif dataset == 'meps-h192':
        return preprocess_meps_h192_dataset()
    else:
        raise ValueError('Invalid dataset name.')
    #end if
#end get_data()

#dataset_list = ['heart-cleveland', 'adult', 'autism', 'german-credit', 'credit-card',
#                'student-entrance', 'student-perf', 'compas',
#                'recidivism-juvenile']
dataset_list= ['german-credit']
for dataset in dataset_list:
    get_dataset(dataset)
#end for
