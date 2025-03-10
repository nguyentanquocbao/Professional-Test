import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from optbinning.scorecard.plots import plot_ks, plot_auc_roc
# Assuming df is your original dataframe with ['customer_id', 'disbursement_date'] as index
# Function to parse and flatten the CIC_DATA
def parse_row_cic(row_in):
    try:
        data = json.loads(row_in).get('NOIDUNG', {})
        result = {}
        
        # Iterate through main sections of the data
        for section_key, section_value in data.items():
            if not isinstance(section_value, dict):
                result[section_key] = section_value
                continue
                
            # Handle second level dictionaries
            for sub_key, sub_value in section_value.items():
                if not isinstance(sub_value, dict):
                    result[sub_key] = sub_value
                    continue
                
                # Handle DONG arrays
                if 'DONG' in sub_value and isinstance(sub_value['DONG'], list):
                    dong_array = sub_value['DONG']
                    for count_i, item in enumerate(dong_array):
                        # Process each item in the DONG array
                        for item_key, item_value in item.items():
                            if not isinstance(item_value, dict):
                                # Direct values, add with index
                                result[f'{sub_key}_{item_key}_{count_i}'] = item_value
                            elif isinstance(item_value, dict) and 'DONG' in item_value and isinstance(item_value['DONG'], list):
                                # Handle nested DONG arrays (like CTLOAIVAY)
                                nested_array = item_value['DONG']
                                for nested_item in nested_array:
                                    for nested_key, nested_value in nested_item.items():
                                        result[f'{sub_key}_{item_key}_{nested_key}_{count_i}'] = nested_value

        return result
    except Exception as e:
        return {'error': str(e)}


aggregrate_input={
    "numeric":[
    'DUNO_12THANG_DUNOTHE',
    'DUNO_12THANG_DUNOVAY',
    'DUNO_12THANG_THANG',
    'DUNO_12THANG_TONGDUNO',
    'DUNO_THETD_HANMUC_THETD',
    'DUNO_THETD_SOTIEN_CHAM_TT',
    'DUNO_THETD_SOTIEN_PHAI_TT',
    'NHOM2_12THANG_TONGDUNO',
    'NOXAU_60THANG_NOXAU_USD',
    'QHTD_CTLOAIVAY_DUNO_THE_USD',
    'QHTD_CTLOAIVAY_DUNO_THE_VND',
    'QHTD_CTLOAIVAY_DUNO_USD',
    'QHTD_CTLOAIVAY_DUNO_VND',
    'QHTD_CTLOAIVAY_NHOM1_USD',
    'QHTD_CTLOAIVAY_NHOM2_USD',
    'QHTD_CTLOAIVAY_NHOM3_USD',
    'QHTD_CTLOAIVAY_NHOM4_USD',
    'QHTD_CTLOAIVAY_NHOM5_USD',
    'QHTD_CTLOAIVAY_NHOM1_VND',
    'QHTD_CTLOAIVAY_NHOM2_VND',
    'QHTD_CTLOAIVAY_NHOM3_VND',
    'QHTD_CTLOAIVAY_NHOM4_VND',
    'QHTD_CTLOAIVAY_NHOM5_VND',
    'QHTD_CTLOAIVAY_NHOMNO',
    'QHTD_CTLOAIVAY_NOXAU_KHAC_USD',
    'QHTD_CTLOAIVAY_NOXAU_KHAC_VND',
    'QHTD_TONG_USD',
    'QHTD_TONG_VND',
    'NOXAU_60THANG_NOXAU_VND'
    ],
    "integer":[
    'DUNO_THETD_SOLUONG_THETD',
    'DUNO_THETD_SONGAY_CHAM_TT',
    ],
    "category":[
    'DUNO_THETD_MATCTD',
    'HDTD_MATCTD',
    'NHOM2_12THANG_MATCTD',
    'NOXAU_60THANG_MATCTD',
    'NOXAU_60THANG_NGAYPS_NOXAU_CUOICUNG',
    'NOXAU_60THANG_NHOMNO',
    'NOXAU_60THANG_NHOMNO_CAONHAT',
    'QHTD_CTLOAIVAY_LOAIVAY',
    'QHTD_MATCTD',
    ],
    "remove_redunt":[
    'DUNO_THETD_NGAYSL',
    'HDTD_NGAYKT_HDTD',
    'DUNO_THETD_TENTCTD',
    'HDTD_NGAYKY_HDTD',
    'HDTD_SOHDTD',
    'HDTD_TENTCTD',
    'NHOM2_12THANG_NGAYSL',
    'NOXAU_60THANG_NGAYSL',
    'NOXAU_60THANG_TENTCTD',
    'QHTD_NGAYSL',
    'QHTD_TENTCTD',
    'NHOM2_12THANG_TENTCTD',
    'NHOM2_12THANG_THANG'
    ]
}

function_mapping={
    "numeric":[min,max,np.mean,sum],
    "integer":[min,max,np.median,sum],
    "category":['count','nunique'],
}

def process_columns(df, aggregrate_input, function_mapping):
    """
    Process columns based on their categories defined in aggregrate_input and apply functions from function_mapping.
    Uses vectorized operations to avoid DataFrame fragmentation.
    
    Args:
        df: The dataframe to process
        aggregrate_input: Dictionary with column types as keys and lists of column prefixes as values
        function_mapping: Dictionary with column types as keys and lists of aggregation functions as values
    
    Returns:
        Processed dataframe with aggregated statistics
    """
    # Add debug logging to check column matching
    # print(f"DataFrame columns before processing: {df.columns.tolist()}")
    
    # Create a list to collect all new columns
    new_columns = {}
    columns_to_drop = []
    
    for column_type, columns_list in aggregrate_input.items():
        if column_type == "remove_redunt":
            # Just collect columns for later removal
            for col_prefix in columns_list:
                columns_to_drop.extend([col for col in df.columns if col.startswith(col_prefix)])
            continue
            
        if column_type in function_mapping:
            functions_to_apply = function_mapping[column_type]
            
            for col_prefix in columns_list:
                # Find all columns that match this prefix
                matching_columns = [col for col in df.columns if col.startswith(col_prefix)]
                if not matching_columns:
                    continue
                
                # Create a temporary dataframe with the matching columns
                temp_df = df[matching_columns].copy()
                
                # Convert columns to appropriate type based on category
                if column_type in ["numeric", "integer"]:
                    for col in matching_columns:
                        # Convert to string first to handle any potential non-string values
                        temp_df[col] = temp_df[col].astype(str)
                        # Replace '-' and empty strings with NaN
                        temp_df[col] = temp_df[col].replace(['-', ''], np.nan)
                        # Convert to numeric
                        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                    
                    if column_type == "integer":
                        temp_df = temp_df.fillna(0).astype(int)
                
                # Apply each function and collect results
                for func in functions_to_apply:
                    if func == "count" or func == 'count':
                        # Count non-NaN values
                        new_columns[f'{col_prefix}_count'] = temp_df.count(axis=1)
                    elif func == "nunique" or func == 'nunique':
                        # Count unique values
                        new_columns[f'{col_prefix}_nunique'] = temp_df.nunique(axis=1)
                    else:
                        # Apply aggregation function vectorized
                        func_name = func.__name__
                        if func == min:
                            new_columns[f'{col_prefix}_{func_name}'] = temp_df.min(axis=1)
                        elif func == max:
                            new_columns[f'{col_prefix}_{func_name}'] = temp_df.max(axis=1)
                        elif func == np.mean:
                            new_columns[f'{col_prefix}_{func_name}'] = temp_df.mean(axis=1)
                        elif func == sum:
                            new_columns[f'{col_prefix}_{func_name}'] = temp_df.sum(axis=1)
                        elif func == np.median:
                            new_columns[f'{col_prefix}_{func_name}'] = temp_df.median(axis=1)
                
                # Add matching columns to the drop list
                columns_to_drop.extend(matching_columns)
    
    # Create final dataframe with all new columns at once to avoid fragmentation
    result_df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        result_df = pd.concat([result_df, new_df], axis=1)
    
    return result_df

# Legacy function kept for backwards compatibility
def process_debt_columns(df, column_prefix, suffix_to_keep=None):
    """
    Process columns with a specific prefix in the dataframe. (Legacy function)
    
    Args:
        df: The dataframe to process
        column_prefix: The prefix of columns to process (e.g., 'debt_loan_')
        suffix_to_keep: If not None, only keep these columns after processing
    
    Returns:
        The processed dataframe with aggregated statistics
    """
    # Extract columns with the specified prefix
    prefix_columns = [col for col in df.columns if col.startswith(column_prefix)]
    
    if not prefix_columns:
        return df
    
    # Convert to proper numeric values
    for col in prefix_columns:
        # First, convert to string to handle any potential non-string values
        df[col] = df[col].astype(str)
        # Replace '-' and empty strings with NaN
        df[col] = df[col].replace(['-', ''], np.nan)
        # Convert to float, errors='coerce' will convert non-numeric values to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate statistics
    df[f'max_{column_prefix[:-1]}'] = df[prefix_columns].max(axis=1)
    df[f'min_{column_prefix[:-1]}'] = df[prefix_columns].min(axis=1)
    df[f'avg_{column_prefix[:-1]}'] = df[prefix_columns].mean(axis=1)
    
    # If suffix_to_keep is specified, keep only those columns
    if suffix_to_keep:
        columns_to_keep = [f"{column_prefix}{suffix}" for suffix in suffix_to_keep]
        columns_to_drop = [col for col in prefix_columns if col not in columns_to_keep]
    else:
        columns_to_drop = prefix_columns
    
    # Drop the original columns
    df.drop(columns=columns_to_drop, inplace=True)
    
    return df

# Apply the parsing function to each row in the CIC_DATA column


# Count unique values in search_x_code columns
def count_search_code_columns(df):
    # Find all columns that match the pattern 'search_*_code'
    search_code_columns = [col for col in df.columns if col.startswith('search_') and col.endswith('_code')]
    
    # Create a new column with the count of unique values
    if search_code_columns:
        # Get the number of unique institutions searched
        df['num_unique_search_institutions'] = df[search_code_columns].nunique(axis=1)
        
        # Count how many searches were performed
        df['num_total_searches'] = df[search_code_columns].notna().sum(axis=1)
    
    return df.drop(columns=search_code_columns)

def remove_redunt(df,prefix,suffix):
    # Find all columns that match the pattern 'search_*_code'
    search_code_columns = [col for col in df.columns if col.startswith(prefix) and col.endswith(suffix)]
    
    # Create a new column with the count of unique values
    if search_code_columns:
        # Get the number of unique institutions searched
        df['num_unique_search_institutions'] = df[search_code_columns].nunique(axis=1)
        
        # Count how many searches were performed
        df['num_total_searches'] = df[search_code_columns].notna().sum(axis=1)
    
    return df.drop(columns=search_code_columns)

def split_data(df, target_column='label', test_size=0.2, random_state=42, max_ratio=0.5, rebalance=True):
    """
    Split data into train and test sets while ensuring that no category 
    in the target variable exceeds the specified maximum ratio (default 50%) 
    in either train or test sets.
    
    Args:
        df: DataFrame containing the data
        target_column: Name of the target column (default: 'label')
        test_size: Proportion of data to include in the test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        max_ratio: Maximum allowed proportion of any single category (default: 0.5)
        rebalance: Whether to rebalance the datasets to enforce max_ratio (default: True)
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Initial check on target distribution
    target_counts = df[target_column].value_counts()
    target_ratios = df[target_column].value_counts(normalize=True)
    
    print(f"Original data distribution:")
    print(target_ratios)
    
    # Function to balance a dataset (either train or test)
    def balance_dataset(data, target, max_ratio):
        # Create a new DataFrame to hold balanced data
        balanced_data = []
        
        # Calculate the target distribution
        class_counts = data[target].value_counts()
        class_ratios = data[target].value_counts(normalize=True)
        
        # Find overrepresented classes
        overrepresented = class_ratios[class_ratios > max_ratio].index.tolist()
        
        if not overrepresented:
            return data  # No balancing needed
            
        # Calculate the target size for overrepresented classes
        # For each overrepresented class, we want: class_size / total_size = max_ratio
        # This means: class_size = max_ratio * total_size
        # But total_size = sum(other_classes) + class_size 
        # So: class_size = max_ratio * (sum(other_classes) + class_size)
        # Solving for class_size: class_size = (max_ratio * sum(other_classes)) / (1 - max_ratio)
        
        # First pass: estimate total without overrepresented classes
        total_underrepresented = sum(class_counts[~class_counts.index.isin(overrepresented)])
        
        # Process each class
        for class_value, count in class_counts.items():
            class_data = data[data[target] == class_value]
            
            if class_value in overrepresented:
                # Max allowed size for this class to meet the max_ratio constraint
                max_size = int((max_ratio * total_underrepresented) / (1 - max_ratio))
                # Downsample the class to the calculated size
                if len(class_data) > max_size:
                    class_data = class_data.sample(max_size, random_state=random_state)
            
            balanced_data.append(class_data)
        
        # Combine all classes
        result = pd.concat(balanced_data).reset_index(drop=True)
        return result
    
    # First split with stratification to maintain the original distribution
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Use sklearn's stratified split initially
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create dataframes for train and test
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Check if balancing is needed
    train_ratios = y_train.value_counts(normalize=True)
    test_ratios = y_test.value_counts(normalize=True)
    
    needs_balancing = (train_ratios > max_ratio).any() or (test_ratios > max_ratio).any()
    
    # Balance both train and test sets if needed and if rebalance is True
    if needs_balancing and rebalance:
        print(f"\nSome categories exceed {max_ratio*100}% in initial split. Balancing datasets...")
        
        # Balance train and test sets separately
        train_df = balance_dataset(train_df, target_column, max_ratio)
        test_df = balance_dataset(test_df, target_column, max_ratio)
        
        # Split features and target again
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
    elif needs_balancing and not rebalance:
        print(f"\nSome categories exceed {max_ratio*100}% in split, but rebalancing is disabled.")
    
    # Validate the final distribution
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    print("\nFinal train set distribut ion:")
    print(train_dist)
    
    print("\nFinal test set distribution:")
    print(test_dist)
    
    # Final check if any category still exceeds max_ratio
    if (train_dist > max_ratio).any() or (test_dist > max_ratio).any():
        print(f"\nNote: Some categories exceed {max_ratio*100}% in train or test set")
        print("Categories exceeding in train:", train_dist[train_dist > max_ratio].index.tolist())
        print("Categories exceeding in test:", test_dist[test_dist > max_ratio].index.tolist())
    else:
        print(f"\nSuccess: No category exceeds {max_ratio*100}% in either train or test sets")
    
    return X_train, X_test, y_train, y_test

def compute_gini_multiclass(build_df):
    
    total_count = build_df.loc['Totals', 'Count'] if 'Totals' in build_df.index else build_df['Count'].sum()
    event_columns = [col for col in build_df.columns if col.startswith('Event_') and not col.startswith('Event_rate')]
    auc_sum = 0
    weight_sum = 0

    # Process each class
    for col in event_columns:
        # Skip totals row for calculations
        df_no_totals = build_df[~build_df.index.isin(['Totals'])]
        class_counts = df_no_totals[col]
        total_class_count = class_counts.sum()
        class_props = class_counts / total_class_count
        total_props = df_no_totals['Count'] / total_count
        class_auc = 0
        cum_class_prop = 0
        cum_total_prop = 0
        sorted_indices = np.argsort(-class_props)
        for i in sorted_indices:
            class_prop = class_props.iloc[i]
            total_prop = total_props.iloc[i]
            class_auc += total_prop * (2 * cum_class_prop + class_prop) / 2
            cum_class_prop += class_prop
            cum_total_prop += total_prop
        weight = total_class_count / total_count
        auc_sum += class_auc * weight
        weight_sum += weight
    auc_weighted = auc_sum / weight_sum if weight_sum > 0 else 0.5
    return 2 * auc_weighted - 1
from optbinning import BinningProcess,Scorecard
selection_criteria = {"iv": {"min": 0.005, 'max':0.5, "strategy": "highest"}}
def find_useful_feasture(X_train,y_train,list_features):
    final_list=[]
    erorr_list=[]
    for i in list_features:
        # Instatiate BinningProcess
        print(i)
        try:
            # Handle missing values in numeric and categorical features
            list_numeric = X_train[[i]].select_dtypes(include=['number']).columns.values
            list_categorical = X_train[[i]].select_dtypes(include=['object', 'category']).columns.values
            # For numeric features, `handle missing values by filling with median
            for col in list_numeric:
                X_train[col] = X_train[col].fillna(0)
                
            # For categorical features, handle missing values by filling with the most frequent value
            for col in list_categorical:
                X_train[col] = X_train[col].fillna(X_train[col].mod)[0]
                
            # Ensure all categorical variables are properly represented
            binning_process = BinningProcess(
                categorical_variables=list_categorical,
                variable_names=[i],
                selection_criteria=selection_criteria,
            )
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import HuberRegressor

            # Initialize the Huber regressor with robust parameters
            rf_model = HuberRegressor( max_iter=200, )
            scaling_method = "min_max"
            scaling_method_data = {"min": 0, "max": 1000}

            # Instatiate and fit Scorecard
            scorecard = Scorecard(
                binning_process=binning_process,
                estimator=rf_model,
                scaling_method=scaling_method,
                scaling_method_params=scaling_method_data,
                intercept_based=False,
                reverse_scorecard=True,
            )

            scorecard.fit(X_train[[i]],y_train)
            final_list.append(i)
        except ValueError as e:
            erorr_list.append(i)
            continue
    return final_list,erorr_list

from sklearn.linear_model import HuberRegressor
def Create_Score_Card(x_train, y_train, final_list):
    """
    Create a scorecard model after converting numeric columns stored as strings to their proper type.
    
    Args:
        x_train: DataFrame with features
        y_train: Series with target variable
        final_list: List of features to include in the model
    
    Returns:
        Fitted Scorecard model
    """
    # Make a copy to avoid modifying original data
    x_train_processed = x_train[final_list].copy()
    
    # Identify and convert numeric columns stored as strings
    for col in x_train_processed.columns:
        # Check if column is object/string type
        if x_train_processed[col].dtype == 'object' or x_train_processed[col].dtype.name == 'category':
            # Try to convert to numeric
            try:
                # First check if it can be converted to numeric
                numeric_series = pd.to_numeric(x_train_processed[col], errors='coerce')
                # Only convert if we don't lose too much data (less than 10% NaN after conversion)

            except Exception as e:
                print(f"Could not convert column {col}: {str(e)}")
                continue
    # Identify truly categorical variables (object/category type that couldn't be converted)
    list_categorical = x_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical variables for binning: {list_categorical}")
    # Create binning process
    binning_process = BinningProcess(
        categorical_variables=list_categorical,
        variable_names=final_list,
        selection_criteria=selection_criteria,
    )
    # Initialize the Huber regressor with robust parameters
    rf_model = HuberRegressor(max_iter=500)
    scaling_method = "min_max"
    scaling_method_data = {"min": 0, "max": 1000} 
    # Instantiate and fit Scorecard
    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=rf_model,
        scaling_method=scaling_method,
        scaling_method_params=scaling_method_data,
        intercept_based=False,
        reverse_scorecard=True,
    )

    # Use the processed dataframe
    scorecard.fit(x_train_processed, y_train)
    return scorecard

def calculate_psi(X_train, X_test, features_list, bins=10, verbose=True):
    """
    Calculate Population Stability Index (PSI) for features between training and test datasets.
    
    PSI = Σ (Actual_% - Expected_%) × ln(Actual_% / Expected_%)
    
    PSI interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change, worth investigating
    - PSI >= 0.2: Significant change, requires attention
    
    Args:
        X_train: Training dataset (DataFrame)
        X_test: Test dataset (DataFrame)  
        features_list: List of features to calculate PSI for
        bins: Number of bins for continuous variables (default: 10)
        verbose: Whether to print PSI values and interpretation (default: True)
        
    Returns:
        DataFrame with PSI values and interpretations for each feature
    """
    import numpy as np
    import pandas as pd
    
    results = []
    
    for feature in features_list:
        if feature not in X_train.columns or feature not in X_test.columns:
            if verbose:
                print(f"Feature '{feature}' not found in one or both datasets. Skipping.")
            continue
            
        # Handle categorical variables
        if X_train[feature].dtype.name == 'category' or X_test[feature].dtype.name == 'category' or \
           X_train[feature].dtype == 'object' or X_test[feature].dtype == 'object':
            
            # Get unique categories from both datasets
            categories = pd.concat([X_train[feature], X_test[feature]]).unique()
            
            # Calculate distributions
            train_dist = X_train[feature].value_counts(normalize=True).reindex(categories).fillna(0.0001)
            test_dist = X_test[feature].value_counts(normalize=True).reindex(categories).fillna(0.0001)
            
        else:
            # For continuous variables, create bins based on training data
            min_val = X_train[feature].min()
            max_val = X_train[feature].max()
            
            # Extend the range slightly to include all test data
            range_extension = (max_val - min_val) * 0.01
            bin_edges = np.linspace(min_val - range_extension, max_val + range_extension, bins + 1)
            
            # Add extreme bins for outliers in test set
            bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])
            
            # Bin the data
            train_counts, _ = np.histogram(X_train[feature], bins=bin_edges)
            test_counts, _ = np.histogram(X_test[feature], bins=bin_edges)
            
            # Calculate distributions (add small value to avoid division by zero)
            train_dist = train_counts / train_counts.sum() + 0.0001
            test_dist = test_counts / test_counts.sum() + 0.0001
        
        # Calculate PSI components
        psi_values = (test_dist - train_dist) * np.log(test_dist / train_dist)
        psi = np.sum(psi_values)
        
        # Determine interpretation
        if psi < 0.1:
            interpretation = "No significant change"
        elif psi < 0.2:
            interpretation = "Moderate change - investigate"
        else:
            interpretation = "Significant change - requires attention"
            
        results.append({
            'Feature': feature,
            'PSI': psi,
            'Interpretation': interpretation
        })
        
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('PSI', ascending=False)
    
    if verbose:
        print("Population Stability Index (PSI) Results:")
        print(results_df)
        
        # Count features by PSI category
        stable = sum(results_df['PSI'] < 0.1)
        moderate = sum((results_df['PSI'] >= 0.1) & (results_df['PSI'] < 0.2))
        significant = sum(results_df['PSI'] >= 0.2)
        
        print(f"\nSummary: {stable} stable features, {moderate} features with moderate change, {significant} features with significant change")
    
    return results_df

def plot_psi_distribution(psi_results, figsize=(12, 6)):
    """
    Plot PSI values as a horizontal bar chart with color-coded interpretation.
    
    Args:
        psi_results: DataFrame with PSI values and interpretations from calculate_psi function
        figsize: Tuple specifying figure dimensions
        
    Returns:
        Matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Sort by PSI value
    df = psi_results.sort_values('PSI')
    
    # Create color mapping based on PSI values
    colors = ['green' if psi < 0.1 else 'orange' if psi < 0.2 else 'red' for psi in df['PSI']]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    bars = ax.barh(df['Feature'], df['PSI'], color=colors)
    
    # Add vertical lines for thresholds
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7)
    
    # Annotate PSI values
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{df["PSI"].iloc[i]:.4f}', va='center')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Stable (PSI < 0.1)'),
        Line2D([0], [0], color='orange', lw=4, label='Moderate Change (0.1 ≤ PSI < 0.2)'),
        Line2D([0], [0], color='red', lw=4, label='Significant Change (PSI ≥ 0.2)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Set plot title and labels
    ax.set_title('Population Stability Index (PSI) by Feature', fontsize=14)
    ax.set_xlabel('PSI Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def catering_score(full_data,name_x,name_y):
    from optbinning import OptimalBinning
    opt2=OptimalBinning(
        name=name_x,
        dtype='numerical',
        solver="cp",
        max_n_bins=5,
        monotonic_trend="ascending",
    )
    opt2.fit(full_data[name_x], full_data[name_y])
    return opt2.binning_table.build()