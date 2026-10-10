import numpy as np
import pandas as pd

COLOANE = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
COLOANE_TEXT = ['protocol_type', 'service', 'flag']

def Standardizare(X):
    medie = np.mean(X, axis=0)
    deviatie = np.std(X, axis=0)
    
    deviatie[deviatie == 0] = 1e-10
    X_std = (X - medie) / deviatie
    
    return X_std, medie, deviatie

def Encodare_train(fisier_train):
    date_trafic = pd.read_csv(fisier_train, names=COLOANE)

    date_trafic_normal = date_trafic[date_trafic['label'] == 'normal.'].copy()    
    date_trafic_normal.drop(columns=['label', 'difficulty'], inplace=True)
    date_trafic_encoded = pd.get_dummies(date_trafic_normal, columns=COLOANE_TEXT, dtype=float)
    coloane_finale = date_trafic_encoded.columns
    
    X = date_trafic_encoded.values.astype('float')
    assert np.isnan(X).sum() == 0, "eroare valori matrice train"
    
    X_std, medie, deviatie = Standardizare(X)
    
    return X_std, medie, deviatie, coloane_finale

def Encodare_real(fisier_real, medie_train, deviatie_train, coloane_train):
    date_real = pd.read_csv(fisier_real, names=COLOANE)
    
    tip_conexiune = date_real['label'].values 
    date_real.drop(columns=['label', 'difficulty'], inplace=True)
    date_real_encoded = pd.get_dummies(date_real, columns=COLOANE_TEXT, dtype=float)
    date_real_aliniat = date_real_encoded.reindex(columns=coloane_train, fill_value=0)
    
    X_real = date_real_aliniat.values.astype('float')
    assert np.isnan(X_real).sum() == 0, "eroare valori matrice real"
    
    X_real_std = (X_real - medie_train) / deviatie_train
    
    return X_real_std, tip_conexiune