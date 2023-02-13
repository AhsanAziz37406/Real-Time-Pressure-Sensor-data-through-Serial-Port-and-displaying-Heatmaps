

%net = resnet50;


% Import sensor data into MATLAB
%sensorData = csvread('D:\PhD Projects\Fall Risk Assesement\left_mid_matlab.csv');
files = dir('D:\PhD Projects\Fall Risk Assesement\*.csv');

% Filter the sensor data to remove noise
[b, a] = butter(4, 0.1); % 4th order Butterworth filter with cutoff frequency of 0.2
filteredData = filtfilt(b, a, files);

% Perform a Fourier transform on the filtered data
fftData = fft(filteredData);

% Analyze the frequency components
frequencyComponents = abs(fftData);

% Plot the frequency components
plot(frequencyComponents);
title('Frequency Components of Sensor Data');
xlabel('Frequency (Hz)');
ylabel('Amplitude');

% Split data into training and testing sets
cv = cvpartition(size(filteredData,1),'HoldOut',0.3);
idx = cv.test;
trainingData = filteredData(~idx,:);
testingData = filteredData(idx,:);

% Train decision tree model
treeModel = fitctree(trainingData(:,1:end-1), trainingData(:,end));

% Make predictions on the testing data
predictions = predict(treeModel, testingData(:,1:end-1));

% Evaluate the model
accuracy = sum(predictions == testingData(:,end))/length(testingData);

disp(['Accuracy: ', num2str(accuracy*100), '%']);

[cmat, order] = confusionmat(testingData(:,end),predictions);

plotconfusion(testingData(:,end),predictions);
