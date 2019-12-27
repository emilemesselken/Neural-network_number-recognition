#!/usr/bin/env python
# coding: utf-8

# In[122]:


# Matrizen
import numpy
# Sigmoid
import scipy.special

# grafik
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

class neuralNetwork:
    
    def __init__ (self, inputnodes, hiddennodes, outputnodes, learningrate):
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # random number between 0.5 & 0.5 in (h/onodes x i/hnodes) matrix
        # w11 w21
        # w12 w22

        #self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5);
        #self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5);

        # Stichprobe aus Normalverteilung - Mittelwert 0.0 - pow ist Standartabweichung
        # (Anzahl der Knoten hoch -0.5 -> Wurzel aus Anzahl der Knoten) - letzter param für Form

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate
        
        # sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    def train(self, inputs_list, targets_list):
        
        # input list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Input -> Hidden - Input mit Gewichten multiplizieren & sigmoid function anwenden
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Hidden -> Output - Hidden mit Gewichten multiplizieren & sigmoid function anwenden
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # o-Fehler berechnen
        output_errors = targets - final_outputs
        
        # h-Fehler backpropagation - w Transformieren!
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # Delta w = lr * Error * o1(1-o1) * o2.T
        # w (ho) update
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # w (ih) update
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    def query(self, inputs_list):
        
        # input list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Input -> Hidden - Input mit Gewichten multiplizieren & sigmoid function anwenden
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Hidden -> Output - Hidden mit Gewichten multiplizieren & sigmoid function anwenden
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[123]:


# PARAMETER
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochen = 5

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[124]:


# Trainingsdaten einlesen - "read-only"
data_file = open("Desktop/makeyourownneuralnetwork-master/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# Scalieren aus 0.01 bis 0.99 & Trainieren
for e in range(epochen):
    for record in data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[125]:


# Testdaten einlesen
test_data_file = open("Desktop/makeyourownneuralnetwork-master/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[126]:


# Gewünschtes Ergebnis Bild anzeigen
# text to numbers - Alle außer erstem element -> [1:]
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', Interpolation='None')


# In[127]:


# Test & Fehler/Performance Evaluation

scorecard = []
counter = 0

for record in test_data_list:
    all_values = record.split(',')
    goal = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    max_val = numpy.argmax(outputs)
    counter += 1
    print("Test:", counter, "  ", max_val, " ", all_values[0])
    if(max_val == goal):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# Output

print()
print(scorecard)
print()
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)


# In[ ]:




