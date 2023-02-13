% Get a list of all the files in the directory
files = dir('D:\PhD Projects\Fall Risk Assesement\sensordata\*.csv');

% Initialize variables to store data and accuracy
allData = [];
accuracies = [];



% Loop through each file
for i = 1:length(files)
    % Read the sensor data from the file
    data = csvread(['sensordata/',files(i).name]);
    

    padSize = -3;
    % Take the absolute value of padSize
    padSize = abs(padSize);
    % Use the abs() function along with padarray function
    allData = padarray(data,padSize);


    if size(allData,2) ~= size(data,2)
    data = padarray(data, [0 size(allData,2)-size(data,2)], 'post');
    end
  
    % concatenate data
    allData = [allData; data];
    
    for t=1:size(data,1)
        for r=1:size(data,2)
            if isnan(data(t,r)) || isinf(data(t,r)) || isempty(data(t,r)) || ischar(data(t,r))
                % handle missing or invalid data
            end
        end
    end
    % Split data into training and testing sets
    cv = cvpartition(size(data,1),'HoldOut',0.3);
    idx = cv.test;
    trainingData = data(~idx,:);
    testingData = data(idx,:);
    
    % Train decision tree model
    treeModel = fitctree(trainingData(:,1:end-1), trainingData(:,end));

    % Make predictions on the testing data
    predictions = predict(treeModel, testingData(:,1:end-1));

    % Evaluate the model
    accuracy = sum(predictions == testingData(:,end))/length(testingData);
    accuracies = [accuracies; accuracy];
    
    % print accuracy
    fprintf('Accuracy for file %s: %.2f%%\n', files(i).name, accuracy*100);
end

% print mean accuracy
meanAccuracy = mean(accuracies);
fprintf('Mean accuracy: %.2f%%\n', meanAccuracy*100);