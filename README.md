# Hand Gesture Detection Using Computer Vision - Comparison between different approaches

## Overview

In this project, I implemented 5 different algorithms that perform the task of Computer Vision on gestures of the human hand, categorizing them into 5 classes. The pre-processing is done in the file **preprocess.py** and the data is stored in an *hdf5* file as a pandas dataframe. Each of the model files performs the task on this data set and after the model is trained, it is stored in a *pickle* file or again *hdf5*. I perform cross-validation on each model in order to see which hyper-parameters produce the best results for each model. Each set of parameters is used multiple times, together with the leave-one-user-out method that would use all the data from only one user (i.e. person whose hand was recorded) as validation set. Then, the model with the highest average validation accuracy on one set of hyper-parameters is chosen as the best model.

Finally, the test data that was set aside during pre-processing is used in the file **test.py** where each model is evaluated on unseen data. All import statements and parameters are stored in the file **utils.py**.

The main packages I used were tensorflow.keras and scikit-learn, as well as numpy and pandas for matrix computations.

## Pre-Processing the Data

The [dataset](http://archive.ics.uci.edu/ml/datasets/Motion+Capture+Hand+Postures) that I used put several markers on a glove of the left hand, and the *x*, *y* and *z* coordinates were recorded. Unfortunately, some of the key-points weren't detected by the camera system. However, each input to a Machine Learning model should have the same number of inputs. Thus, for every sample I extracted the following features

* Minimum and Maximum Value of every *x*, *y* and *z* component (6)
* Average of all *x*, *y* and *z* components respectively (3)
* Standard Deviation of all *x*, *y* and *z* components respectively (3)
* Number of detected Points (1)

<a\>

which gives me 13 features to work with. Afterwards, I normalized each sample to a range between 0 and 1, to not be dependent on the camera resolution or hand size of different samples.

*Note:* I pre-processed the test data in the same step. Normally, this is not something you should do but since each sample is preprocessed individually, it doesn't make a difference when the test set is pre-processed.

## Models Used

### Naive Bayes

The baseline model for this problem is Naive Bayes. This model assumes no dependence between features and therefore cannot be expected to have a high accuracy. However, it is very easy to implement and fast to train, so it serves as a great baseline model. There are no parameters for the cross-validation.

### Support Vector Machine

An SVM tries to find a gap between classes that is as large as possible in order to make the classification as robust as possible. I used a Gaussian kernel and cross-validated on several parameters for the regularization parameter C, which is the loss added to the weights, and $\gamma$ which is the kernel coefficient. The chosen parameters were C=0.1 and &gamma; = 1.

### Neural Network

The most complex of the models is the Neural Network. In fact, running it with different parameters, while cross-validating over 9 users took around 3 hours. I decided on an architecture with two hidden layer with 20 and 10 nodes respectively. The input were 13 nodes and the output has 5 nodes, where each node represents the probability that the sample is from that class. The hyper-parameters that I performed cross-validation on were several values for dropout and regularization since I noticed over-fitting and those two approaches usually counter over-fitting. Dropout is the concept of ignoring a certain percentage of nodes in the layer, therefore making the remaining nodes ore important. The regularization parameter adds a loss onto each weight, therefore reducing the absolute value of the weights which also counters overfitting. The final parameters were Dropout=0.1, Regularization=0.001.

### k Nearest Neighbors

The k Nearest Neighbors algorithm looks at the $k$ nearest points to a given datapoint and is classified depending on the classes of those neighbors. There are mainly two hyper-parameters, one is the number of neighbors that are being evaluated and the other is how the weights are computed. 'Uniform' means that each neighbor is counted equally, while in 'distance' the weight of the neighbors depends on the distance to the datapoint. The values that were chosen are k=11 and weights='distance'.

### Perceptron

I do not expect a Perceptron to perform very well on this data since it is a linear classifier. We do not know how whether the data is linearly separable which is a requirement for a Perceptron to perform well. The hyper-parameters tested for cross-validation were the type of penalty and the penalty value $\alpha$. The best result were achieved with penalty='l2' and &alpha;=0.0001.

## Results

In the following table, it can be seen that the validation accuracy and test accuracy are very similar in most cases, so we did a good job in choosing the right model. The only major difference we can see is in the Perceptron and somewhat in the Naive Bayes model. Since Perceptron is a linear classifier, it depends a lot on how the data is distributed (i.e. is it linearly separable), so there might be a large gap between training, validation and testing.

| Model                  | Validation Accuracy | Test Accuracy |
| ---------------------- | ------------------: | ------------: |
| Naive Bayes            |              79.13% |        70.49% |
| Support Vector Machine |              75.43% |        75.43% |
| Neural Network         |              86.97% |        86.32% |
| k Nearest Neighbors    |              79.49% |        78.80% |
| Perceptron             |              78.07% |        65.24% |

The following image shows the confusion matrix of the Neural Network:

![Confusion Matrix](/Images/confusion_mat.png)

## Notes

There are always ways to improve such a task and here a some ideas:

* **Feature Extraction:** I decided to work with 13 features but of course there are more possibilities for that. For example, we could capture the point that has the largest and smallest value of each of the coordinates, i.e. if x<sub>i</sub> is the smallest values of each x-coordinate, we could also use y<sub>i</sub> and z<sub>i</sub> to get a closer idea of the shape of the hand.
* **Counter Over-fitting:** I noticed that the problem of over-fitting was very apparent. Some ideas on how to counter that were mentioned before but we could do an even better job in that by cross-validating over more parameters.
* **Data Set:** Even though it is understandable why the data set did not record each datapoint, it is hard to make perfect prediction on the features we can extract from the points.  
