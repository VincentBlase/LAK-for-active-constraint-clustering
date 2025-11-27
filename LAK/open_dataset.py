from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA


def open_dataset(name):
    
    enc = LabelEncoder()

    if name == 'DERMATOLOGY':
        # fetch dataset 
        dermatology = fetch_ucirepo(id=33) 
          
        # data (as pandas dataframes) 
        X = dermatology.data.features 
        y = dermatology.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name =='COLUMN_2C':
        # fetch dataset 
        column_2C = fetch_ucirepo(id=212) 
          
        # data (as pandas dataframes) 
        X = column_2C.data.features 
        y = column_2C.data.targets 
        X = X.drop_duplicates(keep='first').dropna()
        y['class'] = y['class'].apply(lambda x : 0 if x == 'Normal' else 1)

    if name =='SONAR':
        # fetch dataset 
        connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151) 
          
        # data (as pandas dataframes) 
        X = connectionist_bench_sonar_mines_vs_rocks.data.features 
        y = connectionist_bench_sonar_mines_vs_rocks.data.targets
        X = X.drop_duplicates(keep='first').dropna()

    if name =='OPTDIGITS389':
        # fetch dataset 
        optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80) 
          
        # data (as pandas dataframes) 
        X = optical_recognition_of_handwritten_digits.data.features 
        y = optical_recognition_of_handwritten_digits.data.targets 
        X = X.iloc[y[y['class'].isin([3,8,9])].index].reset_index(drop=True)
        y = y[y['class'].isin([3,8,9])].reset_index(drop=True)
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'GLASS':
        # fetch dataset 
        glass_identification = fetch_ucirepo(id=42) 
          
        # data (as pandas dataframes) 
        X = glass_identification.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = glass_identification.data.targets 

    if name == 'PARKINSONS' :
        # fetch dataset 
        parkinsons = fetch_ucirepo(id=174) 
          
        # data (as pandas dataframes) 
        X = parkinsons.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = parkinsons.data.targets 

    if name == 'BreastCancerWisconsin':
        # fetch dataset 
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
          
        # data (as pandas dataframes) 
        X = breast_cancer_wisconsin_diagnostic.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = breast_cancer_wisconsin_diagnostic.data.targets 


    if name == 'ECOLI':
        # fetch dataset 
        ecoli = fetch_ucirepo(id=39) 
          
        # data (as pandas dataframes) 
        X = ecoli.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = ecoli.data.targets  

    if name == 'IONOSPHERE':
        # fetch dataset 
        ionosphere = fetch_ucirepo(id=52) 
          
        # data (as pandas dataframes) 
        X = ionosphere.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = ionosphere.data.targets 

    if name == 'HCV':

        # fetch dataset 
        hcv_data = fetch_ucirepo(id=571) 
          
        # data (as pandas dataframes) 
        X = hcv_data.data.features 
        y = hcv_data.data.targets 
        X = X.drop_duplicates(keep='first').dropna()
        X['Sex'] = X['Sex'].apply(lambda x : 0 if x == "m" else 1)

    if name == 'IRIS':
        # fetch dataset 
        iris = fetch_ucirepo(id=53) 
          
        # data (as pandas dataframes) 
        X = iris.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = iris.data.targets


    if name == 'SEGMENTATION':

        # fetch dataset 
        image_segmentation = fetch_ucirepo(id=50) 
          
        # data (as pandas dataframes) 
        X = image_segmentation.data.features 
        y = image_segmentation.data.targets 
        X = X.drop_duplicates(keep='first').dropna()


    if name == 'WINE':
        # fetch dataset 
        wine = fetch_ucirepo(id=109) 
          
        # data (as pandas dataframes) 
        X = wine.data.features
        X = X.drop_duplicates(keep='first').dropna()
        y = wine.data.targets 

    if name == 'YEAST':

        # fetch dataset 
        yeast = fetch_ucirepo(id=110) 
          
        # data (as pandas dataframes) 
        X = yeast.data.features 
        X = X.drop_duplicates(keep='first').dropna()
        y = yeast.data.targets 

    if name == 'HEARTDISEASE':
        # fetch dataset 
        heart_disease = fetch_ucirepo(id=45) 
          
        # data (as pandas dataframes) 
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        X = X.drop_duplicates(keep='first').dropna()


    if name == "SPAMBASE":
        # fetch dataset 
        spambase = fetch_ucirepo(id=94) 
          
        # data (as pandas dataframes) 
        X = spambase.data.features 
        y = spambase.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == "HEPATITIS":
        # fetch dataset 
        hepatitis = fetch_ucirepo(id=46) 
          
        # data (as pandas dataframes) 
        X = hepatitis.data.features 
        y = hepatitis.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'FERTILITY':
        # fetch dataset 
        fertility = fetch_ucirepo(id=244) 
          
        # data (as pandas dataframes) 
        X = fertility.data.features 
        y = fertility.data.targets 
        X = X.drop_duplicates(keep='first').dropna()


    if name == 'HABERMAN':
        # fetch dataset 
        haberman_s_survival  = fetch_ucirepo(id=43) 
          
        # data (as pandas dataframes) 
        X = haberman_s_survival.data.features 
        y = haberman_s_survival.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'SOYBEAN':
        # fetch dataset 
        soybean_large = fetch_ucirepo(id=90) 
          
        # data (as pandas dataframes) 
        X = soybean_large.data.features 
        y = soybean_large.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'MUSK':
        # fetch dataset 
        musk_version_1  = fetch_ucirepo(id=74) 
          
        # data (as pandas dataframes) 
        X = musk_version_1.data.features 
        y = musk_version_1.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name =='TAE':
        X = pd.read_csv('/projects/sig/vblase/data/TAE/tae.data')
        y = np.array(X['3.1'])
        X = X.drop(columns='3.1')
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'SEEDS':
        X = pd.read_csv('/projects/sig/vblase/data/SEEDS/seeds_dataset.txt', sep='\t', on_bad_lines='skip')
        y = np.array(X['1'])
        X = X.drop(columns='1')
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'mfeat_karhunen':
        X = pd.read_csv('/projects/sig/vblase/data/m_feat_kar/mfeat-karhunen.arff', sep=',', skiprows = 154, header=None,on_bad_lines='skip')
        y = np.array(X[64])
        X = X.drop(columns=64)
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'mfeat_zernike':
        X = pd.read_csv('/projects/sig/vblase/data/m_feat_zer/mfeat-zernike.arff', sep=',', skiprows = 137, header=None,on_bad_lines='skip')
        y = np.array(X[47])
        X = X.drop(columns=47)
        X = X.drop_duplicates(keep='first').dropna()

        

    if name == 'PIMA':
        X = pd.read_csv('/projects/sig/vblase/data/PIMA/pima_indians_diabetes.txt', sep=',', on_bad_lines='skip')
        y = np.array(X['1'])
        X = X.drop(columns='1')
        X = X.drop_duplicates(keep='first').dropna()
        

    if name == 'BALANCE':
        # fetch dataset 
        balance_scale = fetch_ucirepo(id=12) 
          
        # data (as pandas dataframes) 
        X = balance_scale.data.features 
        y = balance_scale.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'VEHICLE':
        # fetch dataset 
        statlog_vehicle_silhouettes = fetch_ucirepo(id=149)  
          
        # data (as pandas dataframes) 
        X = statlog_vehicle_silhouettes.data.features 
        y = statlog_vehicle_silhouettes.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'BANKNOTE':
        # fetch dataset 
        banknote_authentication = fetch_ucirepo(id=267) 
          
        # data (as pandas dataframes) 
        X = banknote_authentication.data.features 
        y = banknote_authentication.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'WAVEFORM':
        # fetch dataset 
        waveform_database_generator_version_1 = fetch_ucirepo(id=107) 
          
        # data (as pandas dataframes) 
        X = waveform_database_generator_version_1.data.features 
        y = waveform_database_generator_version_1.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'EEG':
        # fetch dataset 
        eeg_eye_state = fetch_ucirepo(id=264) 
          
        # data (as pandas dataframes) 
        X = eeg_eye_state.data.features 
        y = eeg_eye_state.data.targets 
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'LETTER':
        # fetch dataset 
        letter_recognition = fetch_ucirepo(id=59) 
          
        # data (as pandas dataframes) 
        X = letter_recognition.data.features 
        y = letter_recognition.data.targets 
        X = X.drop_duplicates(keep='first').dropna()


    if name =='RINGS':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/rings.arff"
        data = pd.read_csv(url, skiprows=9, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    if name =='FLAME':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/flame.arff"
        data = pd.read_csv(url, skiprows=10, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'SPHERICAL':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/spherical_6_2.arff"
        data = pd.read_csv(url, skiprows=16, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'LONGSQUARE':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/longsquare.arff"
        data = pd.read_csv(url, skiprows=9, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'JAIN':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/jain.arff"
        data = pd.read_csv(url, skiprows=11, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    if name == 'SQUARE5':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/square5.arff"
        data = pd.read_csv(url, skiprows=7, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    if name == '3MC':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/3MC.arff"
        data = pd.read_csv(url, skiprows=12, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()

    
    if name == '2d-3c-no123':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/2d-3c-no123.arff"
        data = pd.read_csv(url, skiprows=10, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()
        

    if name == '2d-20c-no0':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/2d-20c-no0.arff"
        data = pd.read_csv(url, skiprows=10, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()
        

    if name == '2d-4c-no4':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/2d-4c-no4.arff"
        data = pd.read_csv(url, skiprows=10, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()
        

    if name == 'disk-4500n':
        url ="https://raw.githubusercontent.com/deric/clustering-benchmark/refs/heads/master/src/main/resources/datasets/artificial/disk-4500n.arff"
        data = pd.read_csv(url, skiprows=8, header=None, sep=',')
        X = data.iloc[:,:2]
        y = data[2]
        X = X.drop_duplicates(keep='first').dropna()
        
    y = enc.fit_transform(np.array(y).ravel())
    y = y[X.index]
    
    
    pipe = Pipeline([
                        ('Scalar_2',MinMaxScaler())
                    ])
    
    X = pipe.fit_transform(X)



    return X,y
