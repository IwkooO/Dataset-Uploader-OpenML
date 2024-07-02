import sortinghatinf as shi
import pandas as pd
import nltk
#nltk.download('punkt') 
# NLTK important for sortinghatinf


###################################
## Function to return predictions of features given a dataset
def return_predictions_of_features(df):
    infer_sh = shi.get_sortinghat_types(df) # Get basic sortinghat types

    for col, col_type in zip(df.columns, infer_sh): # If any types are categorical, columns need to be converted to strings 
        if col_type == 'categorical':
            df[col] = df[col].astype(str)
    
    infer_arff, infer_sh = shi.get_feature_types_as_arff(df)
    infer_exp = shi.get_expanded_feature_types(df) # Get expanded feature types. These are the most detailed informative.


    # If type is not-generlizable the arff type should be a STRING.
    # This is not used in the application right now but should be kept in mind if implemented.

    for nr,col in enumerate(infer_exp): 
        if col=="not-generalizable":
            infer_arff[nr]=(infer_arff[nr][0],"STRING")


    return infer_exp, infer_sh, infer_arff

