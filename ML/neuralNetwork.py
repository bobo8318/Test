# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:47:00 2018

@author: My
"""
from numpy import exp,array,random,dot
class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2*random.random((3,1))-1
    
    def __sigmoid_derivative(self,x):
        return x*(1-x)
    
    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
        #print(training_set_inputs.T)
        for iteration in range(number_of_training_iterations):
            output = self.predict(training_set_inputs)#当前权值下预测值
            error = training_set_outputs - output#误差
            print("error:")
            #print(dot(training_set_inputs,self.synaptic_weights))
            adjustment = dot(training_set_inputs.T,error*self.__sigmoid_derivative(output))#权重调整
            print(training_set_inputs.T)
            print(error*self.__sigmoid_derivative(output))
            print(adjustment)
            self.synaptic_weights += adjustment
            
    def __sigmoid(self,x):#激励函数
        return 1/(1+exp(-x))
    
    def predict(self,inputs):
        return self.__sigmoid(dot(inputs,self.synaptic_weights))

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("synaptic weights:")
    print(neural_network.synaptic_weights)
    
    print("input:")
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    print(training_set_inputs)
    
    training_set_outputs = array([[0,1,1,0]]).T
    neural_network.train(training_set_inputs,training_set_outputs,1)
    print("new training synaptic weights:")
    print(neural_network.synaptic_weights)
    
    print("pridect-1,0,0:")
    print(neural_network.predict(array([1,0,0])))
