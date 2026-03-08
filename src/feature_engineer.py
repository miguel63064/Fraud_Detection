import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def categorical_encoding(train, test):
    cat_cols = ['ProductCD', 'card4', 'card6', 'M1','M2','M3','M4','M5',
                'M6','M7','M8','M9','DeviceType','id_12','id_15', 'id_16','id_23','id_27','id_28',
                'id_29', 'id_30', 'id_31', 'id_35','id_36','id_37', 'id_38','id_33','id_34'] 

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[cat_cols] = enc.fit_transform(train[cat_cols])
    test[cat_cols] = enc.transform(test[cat_cols])
    
def combine_features(train, test):
    for df in [train, test]:
        # Combinações de campos que identificam o mesmo utilizador
        df['uid1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        df['uid2'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
        df['uid3'] = df['card1'].astype(str) + '_' + df['P_emaildomain'].astype(str)
    

def aggregate_user_stats(train, test):
    # Para cada utilizador, estatísticas das suas transações
    agg_dict = {
        'TransactionAmt': ['mean', 'std', 'max', 'min'],
        'D1': ['mean', 'std'],
        'C1': ['mean', 'sum'],
    }

    uid_agg = train.groupby('uid1').agg(agg_dict)
    uid_agg.columns = ['uid1_' + '_'.join(c) for c in uid_agg.columns]
    
    for col in uid_agg.columns:
        train[col] = train['uid1'].map(uid_agg[col])
        test[col] = test['uid1'].map(uid_agg[col])


def amt_email_features(train, test):
    for df in [train, test]:
        # log
        df['amt_log'] = np.log1p(df['TransactionAmt'])
        # round value?
        df['amt_is_round'] = (df['TransactionAmt'] % 1 == 0).astype(int)
        # Value vs user mean
        df['amt_vs_uid_mean'] = df['TransactionAmt'] / (df['uid1_TransactionAmt_mean'] + 1)
        # Decimal part of the amount
        df['amt_decimal'] = df['TransactionAmt'] - np.floor(df['TransactionAmt'])  
        # Extrair domínio de topo
        df['P_email_suffix'] = df['P_emaildomain'].str.split('.').str[-1]  # com, net, org...
        # Comprador e destinatário têm o mesmo email?
        df['same_email'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)


def frequency_encoding(train, test):
    cat_cols_not_label = ['P_emaildomain', 'P_email_suffix', 'DeviceInfo', 'uid1', 'uid2', 'uid3', 'R_emaildomain']

    for col in cat_cols_not_label:
        freq = train[col].value_counts(normalize=True)
        
        train[col] = train[col].map(freq)
        
        if col in test.columns:
            test[col] = test[col].map(freq)
            test[col] = test[col].fillna(0)
            
        train[col] =  train[col].fillna(0)
        
def time_features(train, test):
    for df in [train, test]:
        df['day'] = df['TransactionDT'] // (3600 * 24)
        df['day_of_week'] = df['day'] % 7
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['dt_normalized'] = df['TransactionDT'] / df['TransactionDT'].max()
        
def remove_zero_importance(train, test):
    zero_importance = ['V117', 'V113', 'V240', 'V241', 'V305', 'V65', 'V68', 
                       'V88', 'V89', 'V122', 'V104', 'V107', 'V41', 'V325', 
                       'V328', 'id_27', 'V28', 'V27', 'V1', 'V14', 'id_22']
    
    cols_to_drop = [c for c in zero_importance if c in train.columns]
    train.drop(columns=cols_to_drop, inplace=True)
    
    cols_to_drop = [c for c in zero_importance if c in test.columns]
    test.drop(columns=cols_to_drop, inplace=True)
    
    