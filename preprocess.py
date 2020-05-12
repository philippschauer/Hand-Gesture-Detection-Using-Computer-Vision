# Import Statements and parameters
from utils import *

print('--Initializing the Preprocessing--')
print('Loading the data.\n')

# Importing and Pre-Processing the Data
df_train = pd.read_csv('dataset/D_train_large.csv')
df_test = pd.read_csv('dataset/D_test.csv')

y_train = df_train['Class']
y_train.columns = ['Class']

df_train = df_train.drop(['Unnamed: 0'], axis = 1)
df_test = df_test.drop(['Unnamed: 0'], axis = 1)

def preprocess(dataset):
    # Define only the x, y and z data in one dataframe
    x_data = dataset[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']]
    y_data = dataset[['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11']]
    z_data = dataset[['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11']]

    # Computing the mean of each x, y and z coordinate
    mean_x = x_data.mean(axis=1)
    mean_y = y_data.mean(axis=1)
    mean_z = z_data.mean(axis=1)

    # Computing the max value of each coordinate
    max_x = x_data.max(axis=1)
    max_y = y_data.max(axis=1)
    max_z = z_data.max(axis=1)

    # Computing the min value of each coordinate
    min_x = x_data.min(axis=1)
    min_y = y_data.min(axis=1)
    min_z = z_data.min(axis=1)

    # Computing the standard deviatoion of each coordinate
    std_x = x_data.std(axis=1)
    std_y = y_data.std(axis=1)
    std_z = z_data.std(axis=1)

    # Computing the nan values of each point
    nan_vals = x_data.isna().sum(axis=1)

    # Combining everything into a dataframe
    new_dataset = pd.DataFrame([mean_x, mean_y, mean_z, 
                                max_x, max_y, max_z, 
                                min_x, min_y, min_z,
                                std_x, std_y, std_z,]).transpose()


    # Normailze the data with the preprocessing tool from sklearn
    x = new_dataset.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_dataset = pd.DataFrame(x_scaled)

    # Combining the data with the label
    new_dataset = pd.concat([new_dataset, nan_vals, dataset['User'], dataset['Class']], axis=1)
    new_dataset = new_dataset.replace({'User': {10: 3, 0: 4, 11:7}})

    # Labeling the columns
    new_dataset.columns = ['Mean X', 'Mean Y', 'Mean Z', 
                           'Max X', 'Max Y', 'Max Z', 
                           'Min X', 'Min Y', 'Min Z',
                           'STD X', 'STD Y', 'STD Z',
                           'NAN', 'User', 'Class']

    new_dataset = new_dataset.sample(frac=1).reset_index(drop=True)

    return new_dataset


print('Preprocessing the data.\n')
df_train = preprocess(df_train)
df_test = preprocess(df_test)

# Saving the data in an hdf5 file
print('Saving the data in an hdf5 file.\n')
df_train.to_hdf(Config['data_file'], 'train')
df_test.to_hdf(Config['data_file'], 'test')

# The End #
