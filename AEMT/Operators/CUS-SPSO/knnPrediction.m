function [acc, testY_est] = knnPrediction( trainX, trainY, testX , testY, k )
%function [acc] = knnPrediction( training, testing , ker, method)
%return the accuarcy of the 1nn. 
%input:  trainX, testX, the training and test data, each row is an instance
%        trainY, testY, the label, 1, 2, ...
%        k, neighbor size


testY_est = knnpredic(trainX' , trainY' , testX' , k);
acc = 1 - sum(testY_est == testY')/length(testY);