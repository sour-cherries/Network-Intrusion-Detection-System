import numpy as np
import pandas as pd

COLUMNS = [
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
TEXT_COLUMNS = ['protocol_type', 'service', 'flag']

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    std[std == 0] = 1e-10
    X_std = (X - mean) / std
    
    return X_std, mean, std

def encode_train(train_file):
    traffic_data = pd.read_csv(train_file, names=COLUMNS)

    normal_traffic_data = traffic_data[traffic_data['label'] == 'normal.'].copy()    
    normal_traffic_data.drop(columns=['label', 'difficulty'], inplace=True)
    traffic_data_encoded = pd.get_dummies(normal_traffic_data, columns=TEXT_COLUMNS, dtype=float)
    final_columns = traffic_data_encoded.columns
    
    X = traffic_data_encoded.values.astype('float')
    assert np.isnan(X).sum() == 0, "train matrix value error"
    
    X_std, mean, std = standardize(X)
    
    return X_std, mean, std, final_columns

def encode_real(real_file, train_mean, train_std, train_columns):
    real_data = pd.read_csv(real_file, names=COLUMNS)
    
    connection_type = real_data['label'].values 
    real_data.drop(columns=['label', 'difficulty'], inplace=True)
    real_data_encoded = pd.get_dummies(real_data, columns=TEXT_COLUMNS, dtype=float)
    real_data_aligned = real_data_encoded.reindex(columns=train_columns, fill_value=0)
    
    X_real = real_data_aligned.values.astype('float')
    assert np.isnan(X_real).sum() == 0, "real matrix value error"
    
    X_real_std = (X_real - train_mean) / train_std
    
    return X_real_std, connection_type