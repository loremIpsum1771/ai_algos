import csv
import random
import math
import operator

test = True
def normalize(fileData):
	globMax = 0
	globMin = 1000000
	for x in fileData:
		for y in x:
			if y > globMax:
				globMax = y
			elif y < globMin:
				globMin = y;
	for x in fileData:
		for y in x:
			y = (y - globMin)/(globMax- globMin)
			#print "test points after normalize: " + str(y)

def loadDataset(testFile,trainFile):
	testSet = []
	trainingSet = []
	with open(testFile, 'rb') as testCSV:
		testCSV.next()
		testLines = csv.reader(testCSV)
		testSet = list(testLines)
		#for i in range(1,len(testSet)):
		testSet = [[float(y) for y in x] for x in testSet]
		#normalize(testSet)
			# testSet[i] = map(float,testSet[i])
		#print "testSet[i] type: " + str(type(testSet[1][1]))
				
	with open(trainFile, 'rb') as trainCSV:
		trainCSV.next()
		trainingLines = csv.reader(trainCSV)
		trainingSet = list(trainingLines)
		trainingSet = [[float(y) for y in x] for x in trainingSet]
		#normalize(trainingSet)
		# for i in range(1,len(trainingSet)):
		# 	trainingSet[i] = map(float,trainingSet[i])
	#print "test set type:  " + str(type(testSet[0][0]))
	#print "testSet line : " + + str(testSet[0]) + " size: " + str(len(testSet)) + "trainingSet line 1: " + str(trainingSet[0]) + " size: " + str(len(trainingSet)) + '\n'
	i = 0
	# for i in range(0,10):
	# 	print str(testSet[i]) + '\n'
	return testSet, trainingSet

#--------------K-Nearest--------------------------------------------------------------------------------------------------------------------------

def euclideanDistance(instance1, instance2, length):
	distance = 0
	i = 0
	for x in range(1,length):
		distance += pow((instance1[x] - instance2[x]),2)
	

	return math.sqrt(distance)	
		
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance,trainingSet[x],length)
		distances.append((trainingSet[x],dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getResponse(neighbors):
	numOnes = 0
	numZeroes = 0
	for x in range(len(neighbors)):
		if neighbors[x][0] == 1:
			numOnes += 1
		elif neighbors[x][0] == 0:
			numZeroes +=1

		# response = neighbors[x][-1]
		# if response in classVotes:
		# 	classVotes[response] += 1
		# else:
		# 	classVotes[response] = 1

	#sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
	#return sortedVotes[0][0]
	return 1 if numOnes  > numZeroes else 0

def getAccuracy(testSet,predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][0] == predictions[x]:
			correct +=1

	print "Correct: " + str(correct)
	return (correct/float(len(testSet)))* 100.0

#-----------------Perceptron-----------------------------------------------

def predict(row,weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row,weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i +1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

#--------------ID3 Decision Tree------------------------------------------------------------------------------------

def create_decision_tree(data, attributes, target_attr, fitness_func):
	data = data[:]
	vals = [rec[target_attr] for record in data]
	default = majority_value(data, target_attr)

	if not data or (len(attributes)-1) <= 0:
		return default
	elif vals.count(vals[0]) == len(vals):
		return vals[0]
	else:
		best = choose_attribute(data, attributes,target_attr, fitness_func)

		tree = {best:{}}

		for val in get_values(data,best):
			subtree = create_decision_tree(
				get_examples(data, best, val),
				[attr for attr in attributes if attr != best],
				target_attr,
				fitness_func
				)
			tree[best][val] = subtree

	return tree

def entropy(data, target_attr):
	val_freq = {}
	data_entropy = 0.0

	for record in data:
		if (val_freq.has_key(record[target_attr])):
			val_freq[record[target_attr]] += 1.0
		else:
			val_freq[record[target_attr]] = 1.0

	for freq in val_freq.values():
		data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

	return data_entropy

def gain(data, attr,target_attr):

	val_freq = {}
	subset_entropy = 0.0

	for record in data:
		if(val_freq.has_key(record[attr])):
			val_freq[record[attr]] += 1.0
		else:
			val_freq[record[attr]] = 1.0

	for val in val_freq.keys():
		val_prob  = val_freq[val] / sum(val_freq.values())
		data_subset = [record for record in data if record[attr] == val]
		subset_entropy += val_prob * entropy(data_subset, target_attr)
	return (entropy(data, target_attr) - subset_entropy)

	
def main():
	#--------------K-Nearest--------------------------------------------------------------------------------------------------------------------------
	# data1 = [2, 2, 2, 'a']
	# data2 = [4, 4, 4, 'b']
	# distance = euclideanDistance(data1, data2, 3)
	# print 'Distance: ' + repr(distance)
	#prepare data
	trainingSet = []	
	testSet = []
	split = 0.67
	testSet, trainingSet = loadDataset('votes-test.csv','votes-train.csv')
	#print 'Train set: ' + str(len(trainingSet)) + " type of train idx: " + str(type(trainingSet[0][0]))
	#print 'Test set: ' + repr(len(testSet))
	#generate predictions
	# predictions = []
	# k = 7
	# for x in range(len(testSet)):
	# 	neighbors = getNeighbors(trainingSet, testSet[x], k)
	# 	result = getResponse(neighbors)
	# 	predictions.append(result)

	# 	#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	# accuracy = getAccuracy(testSet, predictions)
	# print('Accuracy: ' + repr(accuracy) + '%')

	#-----------------Perceptron-----------------------------------------------

	# test predictions
	# dataset = [[2.7810836,2.550537003,0],
	# 	[1.465489372,2.362125076,0],
	# 	[3.396561688,4.400293529,0],
	# 	[1.38807019,1.850220317,0],
	# 	[3.06407232,3.005305973,0],
	# 	[7.627531214,2.759262235,1],
	# 	[5.332441248,2.088626775,1],
	# 	[6.922596716,1.77106367,1],
	# 	[8.675418651,-0.242068655,1],
	# 	[7.673756466,3.508563011,1]]
	# weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
	# for row in dataset:
	# 	prediction = predict(row, weights)
	# 	print("Expected=%d, Predicted=%d" % (row[-1], prediction))
	l_rate = 0.1
	n_epoch = 10
	predictions = perceptron(trainingSet, testSet,l_rate,n_epoch)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
	#--------------ID3 Decision Tree------------------------------------------------------------------------------------
	test = [1,2,3,4]
	print "initial test: " + str(test)
	test = test[:]
	print "initial test: " + str(test)


main()