# Import Statements and parameters
from utils import *

print('--Initializing the Nayive Bayes Model--')
print('Loading the data.\n')

# Importing the data
df_train = pd.read_hdf(Config['data_file'], 'train')

# This function prepares the data and returns the training and validation
# data, using the leave-one-user-out method
def prepare_data(user_in):

    # Prepare the data for the weights computation
    train_data = df_train.loc[df_train['User'] != user_in]    
    Y_train = train_data['Class']

    userlist = train_data['User']
    weights = userlist.copy()
    
    # Find the user weights so we counter the unbalanced data
    user_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(userlist),
                                                      userlist)

    unique = np.unique(userlist)
    user_weights = dict(zip(unique, user_weights))


    for user in range(Config['num_users']):
        if user != (user_in-1):
            weights = np.where(userlist==user+1, user_weights[user+1], weights) 

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train),
                                                      Y_train)

    # Now, we want to find the class weights to counter the class inbalance
    unique = np.unique(Y_train)
    class_weights = dict(zip(unique, class_weights))  

    for category in range(Config['num_classes']):
        weights = np.where(Y_train==category+1, weights*class_weights[category+1], weights) 

    # Finally, prepare the data
    X_train = train_data.drop(['Class', 'User'], axis = 1)
    val_data = df_train.loc[df_train['User'] == user_in].drop(['User'], axis=1)

    Y_val = val_data['Class']
    X_val = val_data.drop(['Class'], axis = 1)

    return X_train, Y_train, X_val, Y_val, weights


# Fitting the model - Using Gaussian Naive Bayes
print('Training the Naive Bayes Model using the Gaussian Naive Bayes')
all_val_acc = []
for user in range(Config['num_users']):
    x_train, y_train, x_val, y_val, weights = prepare_data(user+1)
    clf = GaussianNB()
    clf.fit(x_train, y_train, sample_weight=weights)
    score_gaussian = clf.score(x_val, y_val)
    all_val_acc.append(score_gaussian)

print('The average validation accuracy for the Gaussian version we can get is: {0:.2f}'.format(100 * np.average(all_val_acc)))
print()


# Fitting the model - Using Multinomial Naive Bayes - does not have great results.
print('Training the Naive Bayes Model using the Multinomial Naive Bayes')
alpha_list = [0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30]
all_accs = []
for alpha_in in alpha_list:
    all_val_acc = []
    for user in range(Config['num_users']):
        x_train, y_train, x_val, y_val, weights = prepare_data(user+1)
        clf = MultinomialNB(alpha=alpha_in)
        clf.fit(x_train, y_train, sample_weight=weights)
        score_cat = clf.score(x_val, y_val)
        all_val_acc.append(score_cat)
    average = np.average(all_val_acc)
    all_accs.append(average)
    # print('The validation accuracy for alpha={0:.2f} is: {1:.2f}'.format(alpha_in, 100 * average))

print('The best validation accuracy for the Multinomial version is: {0:.2f}'.format(100 * np.max(all_accs)))
print()


# Training the chosen model without leaving any user out using the 
X_train, Y_train, _, _, weights = prepare_data(0)
clf = GaussianNB()
clf.fit(X_train, Y_train, sample_weight=weights)

# Save to file in the current working directory
print('Saving the Model into a pickle file.')
with open(Config['bayes_model'], 'wb') as f:
    pickle.dump(clf, f)
f.close()

# The End #
