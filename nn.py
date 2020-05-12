# Import Statements and parameters
from utils import *

print('--Initializing the Neural Network--')
print('Loading the data.\n')

# Importing the data
df_train = pd.read_hdf(Config['data_file'], 'train')

# Define some variables for the neural network
len_seq = Config['num_features']
hidden_1 = Config['nn_hidden_1']
hidden_2 = Config['nn_hidden_2']
num_classes = Config['num_classes']
number_epochs = Config['num_epochs']
batches = Config['nn_batch_size']

# Hyperparameters for cross-validation
dropout_rates = [0.1, 0.2, 0.3]
reg_vals = [0.0001, 0.0005, 0.001]

# This function can do two separate things. # For the cross-validation
# it filters out the given user, trains it on the remaining data and
# validates it on that user.
# After we found the best parameters, we train it with those on the entire
# training data, it stores the model in the directory and plots the architecture
# as well as the graphs for accuracy and loss.
def run_nn(user_in, dropout_rate, reg_val, use_all):

    # Prepare the data for the weights computation
    train_data = df_train.loc[df_train['User'] != user_in]    
    Y_train = train_data['Class'] - 1  # The NN only knows values in [0, 5)

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
    

    # Now, we want to find the class weights to balance the class inbalance
    unique = np.unique(Y_train)
    class_weights = dict(zip(unique, class_weights))  

    for category in range(Config['num_classes']):
        weights = np.where(Y_train==category, weights*class_weights[category], weights) 


    # Finally, prepare the data
    X_train = train_data.drop(['Class', 'User'], axis = 1)
    val_data = df_train.loc[df_train['User'] == user_in].drop(['User'], axis=1)

    Y_val = val_data['Class'] - 1
    X_val = val_data.drop(['Class'], axis = 1)

    # Build the algorithm
    nnet_inputs = Input(shape=(len_seq,), name='Input')

    z = Dense(hidden_1, activation='relu', 
                        kernel_regularizer=regularizers.l2(reg_val), 
                        bias_regularizer=regularizers.l2(reg_val), 
                        name='hidden_Layer_1')(nnet_inputs)

    z = Dropout(dropout_rate)(z)

    z = Dense(hidden_2, activation='relu', 
                        kernel_regularizer=regularizers.l2(reg_val), 
                        bias_regularizer=regularizers.l2(reg_val), 
                        name='Hidden_Layer_2')(z)

    z = Dropout(dropout_rate)(z)

    model_output = Dense(num_classes, activation='sigmoid', 
                                      kernel_regularizer=regularizers.l2(reg_val), 
                                      bias_regularizer=regularizers.l2(reg_val), 
                                      name='Output')(z)

    model = Model(inputs=nnet_inputs, outputs=model_output)

    model.compile(optimizer='adam', 
                  loss=SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])

    # For cross-validation we don't need to save or store anything
    if not use_all:
        results = model.fit(X_train, Y_train, batch_size=batches, 
                                              epochs=number_epochs, 
                                              validation_data=(X_val, Y_val),
                                              sample_weight=weights)

        # Return the validation accuracy after the last epoch
        return results.history['val_accuracy'][-1]

    # After Figuring out what the best configuration is, we train with all training data
    # Now, timing is not an issue on one single run, so we triple the number of epochs
    else:  
        results = model.fit(X_train, Y_train, batch_size=3*batches, 
                                              epochs=3*number_epochs,
                                              sample_weight=weights)

        # Saving the model in an hdf5 file so we can load it for testing
        print('Saving the Model...')
        model.save(Config['nn_model'])
        plot_model(model, to_file='Images/nn.png', show_shapes=True, show_layer_names=True, rankdir="TB", dpi=256)

        # plot the learning curves
        # Note: We don't have curves for validation, since we use the entire dataset for training
        # but since we chose the parameters that perform best on validation data, we know it would
        # perform well on unseen data
        loss = results.history['loss']
        acc = results.history['accuracy']

        epochs = np.arange(len(loss))

        plt.figure()
        plt.plot(epochs, loss, label='Training Loss')
        plt.xlabel('epochs')
        plt.ylabel('Multiclass Cross Entropy Loss')
        plt.title('Loss of the Algorithm')
        plt.legend()
        plt.savefig('Images/NN_Loss.eps')
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of the Neural Network')
        plt.legend()
        plt.savefig('Images/NN_Accuracy.eps')
        plt.show()
        plt.close()


# Training the model on all hyper-parameters
print('Staring the cross-validation.\n')
all_results = np.empty(shape=(Config['num_users'], len(dropout_rates), len(reg_vals)))
 
for dropout in range(len(dropout_rates)):
    for regularization in range(len(reg_vals)):
        for user in range(Config['num_users']):
            k = run_nn(user+1, dropout_rates[dropout], reg_vals[regularization], False)
            all_results[user][dropout][regularization] = k


# Now, we can evaluate the cross-correlation
all_results = np.average(all_results, axis=0)
print('The highest validation accuracy we can get is: {0:.2f}'.format(100 * np.max(all_results)))

# Finding the best parameters
best_val = unravel_index(all_results.argmax(), all_results.shape)

best_dropout_ind = best_val[0]
best_reg_ind = best_val[1]

best_dropout = dropout_rates[best_dropout_ind]
best_reg_val = reg_vals[best_reg_ind]

print('\nWe chose:\ndropout rate: {}\nregularization: {}\n'.format(best_dropout, best_reg_val))

# Running it with the chosen values
run_nn(0, best_dropout, best_reg_val, True)

# The End #
