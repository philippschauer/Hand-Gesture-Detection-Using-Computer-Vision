# Import Statements and parameters
from utils import *

print('--Initializing the Perceptron--')
print('Loading the data.\n')

# Importing the data
df_train = pd.read_hdf(Config['data_file'], 'train')

# Hyper-parameters
penalty_list = ['l2','l1', 'elasticnet']
alpha_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

# This function runs the perceptron algorithm with the hyper-parameters
def run_perceptron(user_in, penalty, alpha):

    # Prepare the data for the weights computation
    train_data = df_train.loc[df_train['User'] != user_in]    

    Y_train = train_data['Class']
    X_train = train_data.drop(['Class', 'User'], axis = 1)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train),
                                                      Y_train)
    

    # Now, we want to find the class weights to balance the class inbalance
    unique = np.unique(Y_train)
    class_weights = dict(zip(unique, class_weights))  

    val_data = df_train.loc[df_train['User'] == user_in] 

    Y_val = val_data['Class']
    X_val = val_data.drop(['Class', 'User'], axis = 1)

    # Computing and fitting the k neighbors classifier
    clf = Perceptron(penalty=penalty, alpha=alpha, class_weight=class_weights)
    clf.fit(X_train, Y_train)

    val_score = clf.score(X_val, Y_val)

    return val_score

# Running the Cross-Validation
print('Initializing the cross-validation.\n')
val_results = np.zeros(shape=(9, len(penalty_list), len(alpha_list)))

for user in range(Config['num_users']):
    for penalty in range(len(penalty_list)):
        for alpha in range(len(alpha_list)):
            val_results[user][penalty][alpha] = run_perceptron(user+1, penalty_list[penalty], alpha_list[alpha])


# Finding the best results from all runs
averaged = np.average(val_results, axis=0) 
print('The highest validation accuracy we can get is: {}'.format(100 * np.max(averaged)))

# Finding the best parameters
best_val = unravel_index(averaged.argmax(), averaged.shape)

best_penalty_ind = best_val[0]
best_alpha_ind = best_val[1]

best_penalty = penalty_list[best_penalty_ind]
best_alpha = alpha_list[best_alpha_ind]

print('chosen alpha:', best_alpha)
print('chosen penalty', best_penalty)

# Training the model on these parameters
Y_train = df_train['Class']
X_train = df_train.drop(['Class', 'User'], axis = 1)

clf = Perceptron(penalty=best_penalty, alpha=best_alpha, class_weight='balanced')
clf.fit(X_train, Y_train)

# Save to file in the current working directory
print('Saving the Model into a pickle file.')
with open(Config['perceptron'], 'wb') as f:
    pickle.dump(clf, f)

# The End #
