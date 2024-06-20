import pandas as pd
from numpy.linalg import inv, pinv, norm, svd, qr
import numpy as np
import random
import scipy.io
import fair_csslib as fair_css
import pandas as pd

import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import scipy.stats as ss
from collections import Counter


def remove_string_columns(df):
    non_string_columns = df.select_dtypes(exclude=['object'])
    return non_string_columns

def one_hot_encode(df):
    return pd.get_dummies(df)


def unit_norm(df, protected_attribute):
    #X = preprocessing.normalize(df, norm='l2', axis=0)
    #df_norm = pd.DataFrame(X, columns=df.columns)
    protected_col = df[protected_attribute]
    other_cols = df.drop(protected_attribute, axis=1)

    # Normalize other columns to have a unit L2 norm
    normalized_other_cols = normalize(other_cols, axis=0, norm='l2')

    # Convert the normalized columns back to a DataFrame
    normalized_other_cols_df = pd.DataFrame(normalized_other_cols, columns=other_cols.columns)

    # Combine the 'sex' column and the normalized columns
    normalized_df = pd.concat([protected_col, normalized_other_cols_df], axis=1)

    return normalized_df

def preprocess_heart_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/heart-disease/cleveland.data")
    print("--------------------------")
    print("Before pre-processing data")
    #instances and attributes
    print('dataset stats', df.shape)
    #sex statistics
    print(df.groupby(['sex'], sort=False).size())
    # missing values
    print('missing values',Counter(df[df.eq('?')].any(axis=1)))
    
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                pass  # or any other action on exception


    #delete all rows with missing value
    df_new = df.replace('?', np.nan).dropna()
    
    df_new.reset_index(drop=True, inplace=True)

    #normalize each column to unit norm
    #df_norm = unit_norm(df_new, 'sex')
    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    #df_new=one_hot_encode(df_new)
     



    print("--------------------------")
    print("After pre-processing data")
    #instances and attributes
    print('dataset stats', df_new.shape)
    #sex statistics
    #print(df_norm.groupby(['sex'], sort=False).size())
    
    grouped = df_new.groupby('sex', sort=False)

    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return A_values,B_values
#end proprocess_heart_dataset()

def preprocess_adult_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/adult/adult.data")

    print("--------------------")
    print("dataset: Adult census -- before preprocessing")
    print(df.shape)
    print(df.groupby(['sex'], sort=False).size())
  
    #encoding labels to integer values
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df=one_hot_encode(df)

    #end for


    #normalize data
    #df_norm = unit_norm(df, 'sex')
    #rounding decimals
    #df_norm = df_norm.round(decimals=5)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df.shape)
    #print(df_norm.groupby(['sex'], sort=False).size())
    
    grouped = df.groupby('sex', sort=False)

    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return A_values,B_values
#end proprocess_adult_dataset()

def preprocess_german_credit_dataset():

    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/german-credit/german.data", delimiter=" ", header=None)

    print("--------------------")
    print("dataset: German credit -- before preprocessing")
    print(df.shape)

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

    #print(df.groupby(['sex'], sort=False).size())
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df=one_hot_encode(df)

    #normalize data
    #df_norm = unit_norm(df_new, 'sex')
    #rounding decimals
    #df_norm = df_norm.round(decimals=5)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df.shape)
    #print(df_norm.groupby(['sex'], sort=False).size())

    grouped = df.groupby('sex', sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return A_values,B_values
#end process_german_credit_dataset()


def preprocess_credit_card_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/credit-card/credit-card.csv")

    print("----------------------------")
    print("Credit card dataset")
    print("instances and attributes:",df.shape)
    df.rename(columns = {'SEX':'sex'}, inplace = True)
    print(df.groupby(['sex'], sort=False).size())
    


    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df=one_hot_encode(df)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df.shape)
    #print(df_norm.groupby(['sex'], sort=False).size())
    
    grouped = df.groupby('sex', sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return A_values,B_values


def preprocess_student_perf_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/student/student-por.csv", delimiter=";")

    print("----------------------------")
    print("Student performance dataset")
    print(df.shape)

    # remove duplicate columns
    df_new = df.copy()
    #df_new = df_new.drop(columns=['first', 'last'])


    #assign strings a numeric value
    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    df_new=one_hot_encode(df_new)
        
    print(df.groupby(['sex'], sort=False).size())
    #end for

    #df_norm = unit_norm(df_new, 'sex')

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df_new.shape)
    #print(df_norm.groupby(['sex'], sort=False).size())

    #write to a csv file
    grouped = df_new.groupby('sex', sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return A_values,B_values



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

def preprocess_compas_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/compas/compas-scores.csv")

    print("-----------------------")
    print("Compas recividism")
    print(df.shape)
    print(df.groupby(['sex'], sort=False).size())
    date_columns=['compas_screening_date', 'dob', 'c_jail_in', 'c_jail_out', 'c_offense_date', 'vr_offense_date', 'v_screening_date', 'screening_date']
    
    df=convert_to_dates_and_calc_days(df, date_columns)
    


    df_new = df[date_columns+['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_charge_degree', 'is_violent_recid', 'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'type_of_assessment', 'decile_score.1', 'score_text']]
    #df_new = df_new.drop(columns=['first', 'last'])
    
    
    df_new = df_new.replace('?', np.nan).dropna()
    df_new.reset_index(drop=True, inplace=True)
    

    
    
    #contains_nan = df_new.isnull().any().any()
    #print(contains_nan)

    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    df_new=one_hot_encode(df_new)
    contains_nan = df_new.isnull().any().any()
    print(contains_nan)    


    # normalize the data
    #df_norm = unit_norm(df_new, 'sex')

    # round to 4 decimal places
    #df_norm = df_norm.round(decimals=5)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df_new.shape)
    #print(df_norm.groupby(['sex'], sort=False).size())

    grouped = df_new.groupby('sex', sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()

    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')
    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    return A_values,B_values

def column_contains_string(df, column_name):
    return df[column_name].apply(lambda x: isinstance(x, str)).any()

def preprocess_communities_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/communities/communities.data")

    print("-----------------------")
    print("Communities and crime")
    print(df.shape)

    # replace unknown values with zero
    df_new=df.drop(columns=['county', 'community', 'communityname', 'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'PolicBudgPerPop'])
    
    df_new = df_new.replace('?', 0)

    # assign numberic values to strings

    #end for

    # introduce two new columns to identify majority white and majority
    # black neighbourhoods
    #df_new['majorityWhite'] = ''
    white_or_else=[]

    rows, cols = df_new.shape
    for r in range(rows):
        if df_new['racePctWhite'][r] >= 0.5:
            #df_new['majorityWhite'][r] = 1.0
            white_or_else.append(1.0)
        else:
            #df_new['majorityWhite'][r] = 0.0
            white_or_else.append(0.0)
        #end if

        #if df_new['racepctblack'][r] >= 0.5:
        #    df_new['majorityBlack'][r] = 1.0
        #else:
        #    df_new['majorityBlack'][r] = 0.0
        #end if
    #end for
    df_new['majorityWhite'] = white_or_else

    

    
    
    #df_new=one_hot_encode(df_new)



    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df_new.shape)
    print(df_new.groupby(['majorityWhite'], sort=False).size())
    
    grouped = df_new.groupby(['majorityWhite'], sort=False)
    A_df = grouped.get_group(1.0).drop(columns=['majorityWhite','racePctWhite', 'racepctblack' ], axis=1)
    B_df = grouped.get_group(0.0).drop(columns=['majorityWhite','racePctWhite', 'racepctblack' ], axis=1)

    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()

    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    
    return A_values,B_values



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
    
def preprocess_recidivism_juvenile_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/recividism-juvenile/recividism-juvenile.csv")

    print("-----------------------")
    print("Recividism juvenile")
    print(df.shape)
    print(df.groupby(['V1_sexe'], sort=False).size())
    
    date_columns=['V10_data_naixement', 'V22_data_fet', 'V30_data_inici_programa', 'V31_data_fi_programa'] #, 'V55_SAVRYdata', 'V117_rein_data_fet_2013', 'V101_rein_data_fet_2015']
    
    df=convert_to_datetimes_and_calc_days(df, date_columns)
    
    df_new=df[['V1_sexe','V2_estranger','V3_nacionalitat', 'V5_edat_fet_agrupat', 'V6_provincia', 'V7_comarca', 'V8_edat_fet','V9_edat_final_programa','V10_data_naixement', 'V11_antecedents','V12_nombre_ante_agrupat','V13_nombre_fets_agrupat','V14_fet','V15_fet_agrupat','V16_fet_violencia','V17_fet_tipus','V19_fets_desagrupats','V20_nombre_antecedents','V21_fet_nombre','V22_data_fet', 'V23_territori', 'V24_programa', 'V25_programa_mesura', 'V27_durada_programa_agrupat', 'V28_temps_inici', 'V29_durada_programa', 'V30_data_inici_programa', 'V31_data_fi_programa']]
    
#    df_new=df.drop(columns=['xxxxxxxx_PERSONALS', 'V4_nacionalitat_agrupat', 'xxxxxxxx_FETS', 'xxxxxxxx_PROGRAMA', 'xxxxxxxxxxxxxx_MRM', 'V26_mesures', 'V32_MRM_resultat', 'V33_MRM_participacio_victima','V34_MRM_tipus','V35_MRM_forma','V36_MRM_negatiu','V37_MRM_conciliacio_victimaexlus','V38_MRM_reparacio_economica','V39_MRM_reparacio_noeconomica','V40_MRM_reparacio_comunitaria','V41_MRM_participacio_trobada','V42_MRM_participacio_notrobada','V43_MRM_iniciativa_parts','V44_MRM_norepa_actitudmen','V45_MRM_noreparacio_decisio_med','V46_MRM_fet_sensevictima','V47_MRM_victima_nolocalitzabl','V48_MRM_victima_rebutja_parti','V49_MRM_victima_vol','V50_MRM_escrit_reflexio','V51_MRM_reparacio_social','V52_MRM_reparacio_activitateducativa','xxxxxxxx_ATM','V53_ATM_tipus','xxxxxxxx_SAVRY', 'V57_@R2_resum_risc_delictes_violents', 'V58_@R3_resum_risc_violencia_centre', 'V59_@R4_resum_risc_sortides_permisos', 'xxxxxxxx_SAVRY_FACTORS_RISC', 'xxxxxxxx_SAVRY_FACTORS_PROTECCIÓ', 'xxxxxxxx_SAVRY_MODEL', 'xxxxxxxx_REINCIDENCIA_2013', 'xxxxxxxx_REINCIDENCIA_2015', 'V54_SAVRYprograma','V55_SAVRYdata','V56_@R1_resum_risc_global_reverse','V60_SAVRY_total_score','V61_SAVRY_historics_total_score', 'V62_SAVRY_socials_total_score', 'V63_SAVRY_individuals_total_score', 'V64_SAVRY_proteccio_total_score', 'V65_@1_violencia_previa','V66_@2_historia_delictes_no_violents','V67_@3_inici_precoç_violencia','V68_@4_fracas_intervencions_anteriors','V69_@5_intents_autolesio_suicidi_anteriors','V70_@6_exposicio_violencia_llar','V71_@7_historia_maltracte_infantil','V72_@8_delinquencia_pares','V73_@9_separacio_precoç_pares','V74_@10_baix_rendiment_escola','V75_@11_delinquencia_grup_iguals','V76_@12_rebuig_grup_iguals','V77_@13_estrés_incapacitat_enfrontar_dificultats','V78_@14_escassa_habilitat_pares_educar','V79_@15_manca_suport_personal_social','V80_@16_entorn_marginal','V81_@17_actitud_negatives','V82_@18_assumpcio_riscos_impulsivitat','V83_@19_problemes_consum_toxics','V84_@20_problemes_maneig_enuig','V85_@21_baix_nivell_empatia_remordiment','V86_@22_problemes_concentracio_hiperactivitat','V87_@23_baixa_colaboracio_intervencions','V88_@24_baix_compromis_escolar_laboral','V89_@P1_impicacio_prosocial','V90_@P2_suport_social_fort','V91_@P3_forta_vinculacio_adult_prosocial','V92_@P4_actitud_positiva_intervencions_autoritat','V93_@P5_fort_compromis_escola_treball','V94_@P6_perseverança_tret_personalitat', 'V95_FACT1mean_ANTISOCIAL','V96_FACT2mean_DINAMICAFAM','V97_FACT3mean_PERSONALITAT', 'V98_FACT4mean_SUPORTSOCIAL', 'V99_FACT5mean_SUSCEPTIBILITAT'])
    

    df_new = df_new.replace('?', np.nan).dropna()
    df_new.reset_index(drop=True, inplace=True)
    
    

    
    #df_new = df_new.drop(columns=['first', 'last'])

    le = LabelEncoder()
    df_new['V1_sexe'] = le.fit_transform(df_new['V1_sexe'])
    df_new=one_hot_encode(df_new)





    #normalize to unit norm
   # df_norm = unit_norm(df_new, 'V1_sexe')

    # round to 4 decimal places
    #df_norm = df_norm.round(decimals=5)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df_new.shape)
    print(df_new.groupby(['V1_sexe'], sort=False).size())

    grouped = df_new.groupby(['V1_sexe'], sort=False)
    A_df = grouped.get_group(1.0).drop('V1_sexe', axis=1)
    B_df = grouped.get_group(0.0).drop('V1_sexe', axis=1)

    
    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    

    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    
    return A_values,B_values


def clean_up_mepsh(df):
    # remove columns that contain strings
    df = df.select_dtypes(exclude=[object])

    # remove columns where more than 90% of the values are zero
    cols = df.columns
    for col in cols:
        if (df[col] == 0).sum() / len(df) > 0.9:
            df = df.drop(col, axis=1)

    return df


def preprocess_meps_h181_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/meps/h181.csv")
    print("-----------------------")
    print("MEPS - h181 ")
    print(df.shape)
    df.rename(columns = {'SEX':'sex'}, inplace = True)
    print(df.groupby(['sex'], sort=False).size())
    
    df=clean_up_mepsh(df)
    
    df_new = df.copy()

    le = LabelEncoder()
    df_new['sex'] = le.fit_transform(df_new['sex'])
    #df_new=one_hot_encode(df_new)

    #normalise to unit norm
    #df_norm = unit_norm(df_new, 'sex')

    #round to 5 decimals
    #df_norm =  df_norm.round(decimals=5)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df_new.shape)
    print(df_new.groupby(['sex'], sort=False).size())

    grouped = df_new.groupby(['sex'], sort=False)
    A_df = grouped.get_group(1.0).drop('sex', axis=1)
    B_df = grouped.get_group(0.0).drop('sex', axis=1)

    
    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()
    
    A_values = normalize(A_values, axis=0, norm='l2')
    B_values = normalize(B_values, axis=0, norm='l2')

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    
    return A_values,B_values
                     
                     
def preprocess_meps_h192_dataset():
    df = pd.read_csv("/scratch/cs/dmg/fair-CSS-datasets/meps/h192.csv")
    print("-----------------------")
    print("MEPS - h192 ")
    print(df.shape)
    df.rename(columns = {'SEX':'sex'}, inplace = True)
    print(df.groupby(['sex'], sort=False).size())

    df_new = df.copy()

    #assign strings a numeric value
    le = LabelEncoder()
    for col in df_new.columns:
        if df_new[col].dtype == object:
            df_new[col] = le.fit_transform(df_new[col])
    #end for

    #normalise to unit norm
    df_norm = unit_norm(df_new, 'sex')

    #round to 5 decimals
    df_norm =  df_norm.round(decimals=5)

    # print stats after preprocessing
    print("---------------------")
    print("After normalizing data")
    print(df_norm.shape)
    print(df_norm.groupby(['sex'], sort=False).size())

    grouped = df_norm.groupby(['sex'], sort=False)
    A_df = grouped.get_group(2).drop('sex', axis=1)
    B_df = grouped.get_group(1).drop('sex', axis=1)

    
    # Convert remaining columns to numpy arrays
    A_values = A_df.to_numpy()
    B_values = B_df.to_numpy()

    #rounding decimals
    A_values = A_values.round(decimals=5)
    B_values = B_values.round(decimals=5)
    
    return A_values,B_values
