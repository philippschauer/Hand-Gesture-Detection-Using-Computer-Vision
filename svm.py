# Import Statements and parameters
from utils import *

print('--Initializing the Support Vector Machine--')
print('Loading the data.\n')

df_train = pd.read_hdf(Config['data_file'], 'train')

# Parameters for the Cross-Validation
num_gamma = Config['num_gamma']
num_c = Config['num_c']
# Values for those hyper-parameters are equidistant on a log-scale
values_for_gamma = np.logspace(-1, 2, num=num_gamma)
values_for_c = np.logspace(-1, 2, num=num_c)

# Function that computes the validation accuracy 
# of the Cross-Validation
def cross_correlation(user_in, gamma_in, C_in):
    val_data = df_train.loc[df_train['User'] == user_in].drop(['User'], axis=1)
    train_data = df_train.loc[df_train['User'] != user_in].drop(['User'], axis=1)

    Y_train = train_data['Class']
    X_train = train_data.drop(['Class'], axis = 1)

    Y_val = val_data['Class']
    X_val = val_data.drop(['Class'], axis = 1)

    # Compute class weights
    unique, counts = np.unique(Y_train, return_counts=True)
    counter = dict(zip(unique, counts))

    clf = svm.SVC(gamma=gamma_in, C=C_in, kernel='rbf', class_weight=counter)
    clf.fit(X_train, Y_train)

    k = clf.score(X_val, Y_val)

    return k

# Running the Algorithm for all values of c and gamma
# while cross-validating between different users.
print('Initializing the cross-validation.\n')
all_results = np.empty(shape=(Config['num_users'], num_c, num_gamma))

for gamma in range(num_gamma):
    g_in = values_for_gamma[gamma]
    for c in range(num_c):
        c_in = values_for_c[c]
        for user in range(9):
            k = float(cross_correlation(user+1, g_in, c_in))
            all_results[user][c][gamma] = k

# Finding the best accuracy
all_results = np.average(all_results, axis=0)
print('The highest validation accuracy we can get is: {}'.format(100 * np.max(all_results)))

# Finding the best parameters
best_val = unravel_index(all_results.argmax(), all_results.shape)

best_c_ind = best_val[0]
best_g_ind = best_val[1]

best_c = values_for_c[best_c_ind]
best_g = values_for_gamma[best_g_ind]

print('The chosen C:', best_c)
print('The chosen gamma:', best_g)

# Running the SVM algorithm with the best values for gamma and c
Y_train = df_train['Class']
X_train = df_train.drop(['Class', 'User'], axis = 1)

unique, counts = np.unique(Y_train, return_counts=True)
counter = dict(zip(unique, counts))

# Fit the model with these best parameters
clf = svm.SVC(gamma=best_g, C=best_c, kernel='rbf', class_weight=counter)
clf.fit(X_train, Y_train)

# Save to file in the current working directory
print('\nSaving the SVM model')
with open(Config['svm_model'], 'wb') as f:
    pickle.dump(clf, f)

# The End #
