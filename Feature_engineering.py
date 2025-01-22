import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def process_data(data_path):
    data = pd.read_csv(data_path)

    data.drop(["RowNumber", "CustomerId", "Surname"], inplace = True, axis = 1)
    
    onehot = OneHotEncoder()
    encoded_data = onehot.fit_transform(data[["Geography"]]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns = onehot.get_feature_names_out(["Geography"]))
    
    data.drop("Geography", axis = 1, inplace = True)
    fin_data = pd.concat([data, encoded_df], axis = 1)

    label_enc = LabelEncoder()
    fin_data["Gender"] = label_enc.fit_transform(fin_data["Gender"])

    return fin_data