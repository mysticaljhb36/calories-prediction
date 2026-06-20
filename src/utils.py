from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import sys
import joblib

from paths import MODELS_DIR




def transform_imputer(func):

    """

    Decorator to apply iterative imputation to numeric columns.

 

    Args:

        func: Function to be wrapped.

 

    Returns:

        Wrapped function.

    """

    def imputer_wrapper(*args, **kwargs):      

        try:    

            imputer = IterativeImputer(

                estimator=RandomForestRegressor(

                                             n_estimators=25,

                                             min_samples_split=3,

                                             max_features='log2',

                                             n_jobs=-1,

                                             random_state=10,

                                             verbose=0

                                            )

                                       )

            X = func(*args, **kwargs)

            categorical_columns=(

                                 X.select_dtypes(

                                 include=['object',

                                          'datetime64[ns]'])

                                 .columns.tolist()

                                )

            numerical_columns=(

                               X.select_dtypes(

                               include=['uint8',                                    

                                        'int64',

                                        'int32',

                                        'float64',

                                        'float32'])

                               .columns.tolist()

                               )   

            Ximputer = np.abs(

                              imputer.fit_transform(

                              X[numerical_columns])

                              )          

            Xframed = pd.DataFrame(

                                   Ximputer,

                                   index=X.index,

                                   columns=numerical_columns

                                   )

            return (

                    pd.concat(

                        [X[categorical_columns]

                        .reset_index(drop=True),

                        Xframed.reset_index(drop=True)],

                        axis=1)

                    ) 

        

        except ValueError:

            raise ValueError(f"Unsupported string/complex data found -"\

                              f" {sys.exc_info()[1]}")

        except Exception:

            raise

    return imputer_wrapper








def save_model(model, filename: str) -> None:
    """
    Save a trained model to the models directory.

    Args:
        model: Trained model object.
        filename: Name of the model file (e.g. 'random_forest.pkl').
    """
    filepath = MODELS_DIR / filename

    joblib.dump(model, filepath)

    print(f"Model saved to: {filepath}")
    
    





def load_model(filename: str):
    """
    Load a trained model from the models directory.

    Args:
        filename: Name of the model file.

    Returns:
        Loaded model object.
    """
    filepath = MODELS_DIR / filename

    return joblib.load(filepath)


