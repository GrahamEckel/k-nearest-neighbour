import numpy as np

# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('...column_3C.dat', converters={6: lambda s: labels.index(s)} )

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]

###Creating the individual functions
def squared_dist(x,y):
    #sum of squared differences
    return np.sum(np.square(x-y))

def find_NN(x):
    #compute the distances from x to every row in the train data
    distances = [squared_dist(x, trainx[i,]) for i in range(len(trainy))]
    #return the index # of the minimum value in the 'distances' array
    return np.argmin(distances)

def NN_classifier(x):
    #Get the index of the the nearest neighbor
    index = find_NN(x)
    #Get the class of the index
    return trainy[index]

#testing the result of our invidiual functions/components as a baseline
test_predictions = [NN_classifier(testx[i,]) for i in range(len(testy))]
print(test_predictions)


###Putting them all together and outputting an array of classifications
def NN_L2(trainx, trainy, testx):
    IndexArray = []
    for x in range(len(testx)):
        Distance = [np.sum((testx[x,]-trainx[i,])**2) for i in range(len(trainy))]
        Index = np.argmin(Distance)
        Label = trainy[Index]
        IndexArray.append(Label)
    return IndexArray

testy_L2 = NN_L2(trainx, trainy, testx)

#matches our baseline
print(testy_L2)
