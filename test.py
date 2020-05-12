# Import Statements and parameters
from utils import *

print('--Initializing the Testing--')
print('Loading the data.\n')

# Set true whichever model we want to test
test_bayes = True
test_svm = True
test_nn = True
test_knn = True
test_per = True

# Importing the test data
df_test = pd.read_hdf('df.hdf5', 'test')

Y_test = df_test['Class']
X_test = df_test.drop(['Class', 'User'], axis = 1)


# Function that plots the confusion matrix. Takes in the numpy array
# of the matrix plus the title of the plot and the short name for the 
# savefig command, e.g. the key for the Neural Network would be 'nn'.
# Plots the matrix and stores it in the Image Folder
def create_confusion_matrix(confusion_mat, model, model_short):

    # Normalize the confusion matrix to get percentages
    confusion_mat = confusion_mat/confusion_mat.sum(axis=1)[:,None]

    print(confusion_mat)

    axis = [1, 2, 3, 4, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_mat, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+axis)
    ax.set_yticklabels(['']+axis)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix of the {}'.format(model))
    plt.savefig('Images/confusion_{}.eps'.format(model_short))
    plt.show()
    plt.close()


####################
# Test Naive Bayes #
####################

def run_naive_bayes():

    print('\n---Testing the Naive Bayes Model---')
    print('Importing the Model...')
    with open(Config['bayes_model'], 'rb') as file:
        bayes_model = pickle.load(file)

    print('Computing Predictions...')
    score = bayes_model.score(X_test, Y_test)
    print("Accuracy on the test set: {0:.2f}%\n".format(100 * score))

    Y_pred = bayes_model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    create_confusion_matrix(confusion_mat, 'Naive Bayes Model', 'bayes')


if test_bayes:
    run_naive_bayes()


######################
# Test the SVM model #
######################

def run_svm_model():

    print('\n---Testing the SVM Model---')
    print('Importing the Model...')
    with open(Config['svm_model'], 'rb') as file:
        svm_model = pickle.load(file)

    print('Computing Predictions...')
    score = svm_model.score(X_test, Y_test)
    print("Accuracy on the test set: {0:.2f}%\n".format(100 * score))

    Y_pred = svm_model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    create_confusion_matrix(confusion_mat, 'Support Vector Machine', 'svm')


if test_svm:
    run_svm_model()


#######################
# Test Neural Network #
#######################

def run_neural_network():

    print('\n---Testing the Neural Network---')
    print('Importing the Model...')
    neural_network = load_model(Config['nn_model'])

    print('Computing Predictions...')
    results = neural_network.evaluate(X_test, Y_test-1, batch_size=128)
    print("Accuracy on the test set: {0:.2f}%\n".format(100 * results[1]))

    Y_pred = neural_network.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    confusion_mat = tf.math.confusion_matrix(Y_test-1, Y_pred)
    confusion_mat = np.array(confusion_mat)

    create_confusion_matrix(confusion_mat, 'Neural Network', 'nn')


if test_nn:
    run_neural_network()


####################
# Test K Neighbors #
####################

def run_knn():
 
    print('\n---Testing the k Nearest Neighbors Model---')
    print('Importing the Model...')
    with open(Config['neighbors'], 'rb') as file:
        kn_model = pickle.load(file)

    print('Computing Predictions...')
    score = kn_model.score(X_test, Y_test)
    print("Accuracy on the test set: {0:.2f}%\n".format(100 * score))

    Y_pred = kn_model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, Y_pred)

    create_confusion_matrix(confusion_mat, 'k Nearest Neighbors Algorithm', 'knn')


if test_knn:
    run_knn()   


###################
# Test Perceptron #
###################

def run_perceptron():
 
    print('\n---Testing the Perceptron Model---')
    print('Importing the Model...')
    with open(Config['perceptron'], 'rb') as file:
        perceptron = pickle.load(file)

    print('Computing Predictions...')
    score = perceptron.score(X_test, Y_test)
    print("Accuracy on the test set: {0:.2f}%\n".format(100 * score))

    Y_pred = perceptron.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, Y_pred)

    create_confusion_matrix(confusion_mat, 'Perceptron', 'per')


if test_per:
    run_perceptron()   

# The End #
