%SVM Code
output = evalc('disp(8)');
% Read in the data file
D = imread('Linear Seperable Data.bmp');
D = D(:,:,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use this function to select out the black/white dots
% The first line separates class 1 (black dots)
% The second line separates class 2 (white dots)
[C1X, C1Y] = ind2sub(size(D), find(D == 0));
[C2X, C2Y] = ind2sub(size(D), find(D == 255));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following four lines of code take the immense number of samples and
% randomly selects 1 samples from every 30 and only uses that ratio to cut
% the data set down to 1/30th the size 
C1X = C1X(1:20:end,:); 
C1Y = C1Y(1:20:end,:);
C2X = C2X(1:20:end,:);
C2Y = C2Y(1:20:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The next four lines take the now split data and plots them
figure(1)
hold on;
plot(C1X, C1Y, '*');
plot(C2X, C2Y, 'D');
title('DATA');
hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following two lines take the split x and y matrices for each class
% and put them together so that there now exists a matrix that holds both
% the x and y data for each class
% Each data set is retried and then randomly shuffled to increase result
% effectiveness
Class1Data = [C1X C1Y];
Class1Data = Class1Data(randperm(size(Class1Data,1)),:);
Class2Data = [C2X C2Y];
Class2Data = Class2Data(randperm(size(Class2Data,1)),:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Then we need a matrix that contains all data points from both classes so
% we create this next matrix.
% Each data set is retried and then randomly shuffled to increase result
% effectiveness
Data = [C1X C1Y ; C2X C2Y];
Data = Data(randperm(size(Data,1)),:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We need a matrix of all ones for the labels for class one so we create a
% matrix of the same size as the class one matrix filled with +1
Class1Label = ones(size(C1X));
% We do the same thing for class two but this matrix is filled with all
% negative ones
Class2Label = -ones(size(C2X));
% We combine these two matrices to create a matrix with one column where
% the top half is positive ones and the bottom half is negative ones
Labels = [Class1Label ; Class2Label];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Since we are doing the three fold cross validation the data needs to be
% split into a testing bunch and a training bunch. Each bunch should be 1/3
% the data for testing and 2/3 the data for training.
C1TwoThirds = ceil(length(Class1Data)*.66);
C1OneThird = length(Class1Data) - C1TwoThirds;
C2TwoThirds = ceil(length(Class2Data)*.66);
C2OneThird = length(Class2Data) - C2TwoThirds;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fold 1 (Top 1/3 Testing, Bottom 2/3 Training)
% Goal:   Now we need to seperate the acquired data into matrices to make the
%         testing and training data useable, the testing data should be the
%         top 1/3 of the data with the remaining being training
% Step 1: Create a matrix the same size as 2/3 the size of class 1, this is the
%         matrix we will use to train
% Step 2: Create a matrix the same size as 1/3 the size of class 1, this is the
%         matrix we will use to test

% Step 1 Below:
C1F1Train = zeros(C1TwoThirds, 2); 
for i = 1 : C1TwoThirds
    for j = 1 : 2 % We know there is an x and y so we need both
        % Take the data from C1 full matrix and enter into Class 1 train matrix
        C1F1Train(i, j) = Class1Data(i, j);
    end
end

% Step 2 Below:
C1F1Test = zeros(C1OneThird, 1);
for i = 1 : C1OneThird
    for j = 1 : 2 % We know there is an x and y so we need both
        % Take the data from C1 full matrix and enter into Class 1 test matrix
        C1F1Test(i, j) = Class1Data(i, j);
    end
end
% Now that we have two new matrices, the testing and training matrices for
% class one we must repeat this process for class two.
C2F1Train = zeros(C2TwoThirds, 2);
for i = 1 : C2TwoThirds
    for j = 1 : 2 % We know there is an x and y so we need both
        % Take the data from C2 full matrix and enter into Class 2 train matrix
        C2F1Train(i, j) = Class2Data(i, j);
    end
end

C2F1Test = zeros(C2OneThird, 1);
for i = 1 : C2OneThird
    for j = 1 : 2 % We know there is an x and y so we need both
        % Take the data from C2 full matrix and enter into Class 2 test matrix
        C2F1Test(i, j) = Class2Data(i, j);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fold 2 (Middle 1/3 testing, Top and Bottom 1/3 Training)
% Goal:   Now we need to seperate the acquired data into matrices to make the
%         testing and training data useable, this one is a little tricky as
%         you need to get the center 1/3 of the data out and use that for
%         testing.
% Step 1: This part is tricky because you need to take two seperate 1/3
%         entire set, you need 1/3 from the top and 1/3 from the bottom.
%         The problem is if the sample is an odd number one of the 1/3 will
%         need to have an extra sample so we dont have a problem
% Step 2: Create a matrix the same size as 2/3 the size of class 1, this is the
%         matrix we will use to train
% Step 3: Create a matrix the same size as 1/3 the size of class 1, this is the
%         matrix we will use to test

% Step 1 Below:
% If 2/3 the number of samples in class one is an odd number do this if
% statement
if mod(C1TwoThirds, 2) == 1 
    % If the number is odd that means one of the 1/3 "sets" will be
    % required to have one more sample than the other 1/3
    C1TrainSetOne = ceil(C1TwoThirds/2); 
    C1TrainSetTwo = C1TwoThirds - C1TrainSetOne;
end
% If 2/3 the number of samples in class one is an even number do this if
% statement            
if mod(C1TwoThirds, 2) == 0
    % If the number is even then both the top and bottom 1/3s will have the
    % same number of samples
    C1TrainSetOne = C1TwoThirds / 2;
    C1TrainSetTwo = C1TwoThirds / 2;
end
% Repeat the same steps as above for class 2
% If 2/3 the number of samples in class two is an odd number do this if
% statement   
if mod(C2TwoThirds, 2) == 1
    C2TrainSetOne = ceil(C2TwoThirds/2);
    C2TrainSetTwo = C2TwoThirds - C2TrainSetOne;
end
% If 2/3 the number of samples in class two is an even number do this if
% statement              
if mod(C2TwoThirds, 2) == 0
    C2TrainSetOne = C2TwoThirds / 2;
    C2TrainSetTwo = C2TwoThirds / 2;
end

% Step 2 Below:
% Now that we have accurately mapped the sizes needed of the testing and
% training matrices for fold 2 we need to populate these matrices with the
% correct data from the original class 1 and 2 matrices.
C1F2Train = zeros(C1TwoThirds, 2);
% Set the first 1/3 to training data for class 1
for i = 1 : C1TrainSetOne
    for j = 1 : 2
        C1F2Train(i, j) = Class1Data(i, j);
    end
end

% Set an index that allow you to go to the next row to continue population
% of class one training set in the next for loop. Currently set the index
% to the amount of samples already in the training set plus one.
index = C1TrainSetOne + 1;
% For i equals the length of the entire data set for class 1 minus the size of
% the size of 1/3 of the entire set plus 1. So basically it is
% [2/3 class one + 1] to the length of the whole sample. i.e. from the 2/3
% point to the bottom of the list giving you the bottom 1/3 of the data
for i = (length(Class1Data) - C1TrainSetTwo) + 1 : length(Class1Data)
    for j = 1 : 2
        % instead of i , we use the index value so that we can not overrite
        % the existing data already in the matrix
        C1F2Train(index, j) = Class1Data(index, j);
    end
    % Index is the current size of the training set being updated as each one is added
    index = index + 1;
end

% Populate the testing data next which will be the remaining middle 1/3 of
% the data since the top and bottom 1/3 were just extracted.
C1F2Test = zeros(C1OneThird, 1);
% Create another index that will be used to start at the correct position
% in the data. We want to start at the number after the 1/3 mark. This is
% where we will get the middle 1/3 out into a new array for testing.
C1F2RowIndex = C1OneThird + 1;
for i = 1 : C1OneThird
    for j = 1 : 2
        C1F2Test(i, j) = Class1Data(C1F2RowIndex, j);
    end
    % Again we increment to keep us in the middle 1/3 of the data
    C1F2RowIndex = C1F2RowIndex + 1;
end

%Initialize the class two fold two train array and fill with zeros
C2F2Train = zeros(C2TwoThirds, 2);
% Set the first 1/3 to class two train set
for i = 1 : C2TrainSetOne
    for j = 1 : 2
        C2F2Train(i, j) = Class2Data(i, j);
    end
end

% Create an index like before
index = C2TrainSetOne + 1;
% For i = 2/3 class 2 size + 1 to the size of class 2, so to the end.
% Add the bottom 1/3 to the existing class 2 training matrix
for i = (length(Class2Data) - C2TrainSetTwo) + 1 : length(Class2Data)
    for j = 1 : 2
        C2F2Train(index, j) = Class2Data(index, j);
    end
    % Increment the position index
    index = index + 1;
end

% Zero matrix 1/3 the size of class 2 by 1
C2F2Test = zeros(C2OneThird, 1);
% Index like before
C2F2RowIndex = C2OneThird;
for i = 1 : C2OneThird
   for j = 1 : 2
       % Collect the first 1/3 of the data from class two and store in the
       % new class 2 test array
       C2F2Test(i, j) = Class2Data(C2F2RowIndex, j);
   end
   % Increment position index
   C2F2RowIndex = C2F2RowIndex + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fold 3 (Top 2/3 training, Bottom 1/3 testing)
% Goal:   Now we need to take the final bottom 1/3 of the class data as
%         our testing data and the rest as training data
% Step 1: Create a matrix the same size as 2/3 the size of class 1, this is the
%         matrix we will use to train
% Step 2: Create a matrix the same size as 1/3 the size of class 1, this is the
%         matrix we will use to test
% Step 1 Below:
C1F3Train = zeros(C1TwoThirds, 2);
% Set the top 2/3 of the data to the class 1 fold 3 training set
for i = 1 : C1TwoThirds
   for j = 1 : 2
       C1F3Train(i, j) = Class1Data(i, j);
   end
end

C1F3Test = zeros(C1OneThird, 1);
% Set the bottom 1/3 to class 1 fold three testing
for i = C1TwoThirds + 1 : size(Class1Data, 1)
   for j = 1 : 2
       C1F3Test(i, j) = Class1Data(i, j);
   end
end

C2F3Train = zeros(C2TwoThirds, 2);
% Set the top 2/3 of the data to the class 2 fold 3 training set
for i = 1 : C2TwoThirds
   for j = 1 : 2
       C2F3Train(i, j) = Class2Data(i, j);
   end
end

C2F3Test = zeros(C2OneThird, 1);
% Set the bottom 1/3 to class 1 fold three testing
for i = C2TwoThirds + 1 : size(Class2Data, 1)
   for j = 1 : 2
       C2F3Test(i, j) = Class2Data(i, j);
   end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOW THAT WE HAVE ALL OF OUR NEEDED CALCULATION MATRICES ALL DIVIDED OUT %
% INTO THE CORRECT SETS OF TESTING AND TRAINING WE CAN NOW START THE      %
% ACTUAL SVM EXERCISE.                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setting a kernel index that will run the entire exercise with one version
% of a particular kernel and then rerun the entire exercise with a
% different kernel
for KernelIndex = 1 : 2
    % Create a for loop from 1 to 3 so you can change to each different c
    % value 10, 100, 1000
    for z = 1 : 3
        % Create a final for loop that iterates which fold to do
        for foldNum = 1 : 3           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % A series of if statements that will check which fold the
            % program is currently on and do a couple things, merge the two
            % classes individual test and train datasets and combine them into a
            % single matrix containing both sets of data, create two labels
            % matrices that have the same number of rows as the size of the
            % classes testing or training data sets. Fill the top 1/2 with
            % 1 and the bottom half with -1.
            % This if stmt is only done because if you are in fold two the ceil
            % function is used which could change the size of the training
            % data matrix so we have to do this seperately for each fold.
            if foldNum == 1
                TrainData = [C1F1Train ; C2F1Train];
                TestData = [C1F1Test ; C2F1Test];
                Labels = [ones(size(C1F1Train, 1), 1) ; -ones(size(C2F1Train, 1), 1)];
                TestLabels = [ones(size(C1F1Test, 1), 1) ; -ones(size(C2F1Test, 1), 1)];
            elseif foldNum == 2
                TrainData = [C1F2Train ; C2F2Train];
                TestData = [C1F2Test ; C2F2Test];
                Labels = [ones(size(C1F2Train, 1), 1) ; -ones(size(C2F2Train, 1), 1)];
                TestLabels = [ones(size(C1F2Test, 1), 1) ; -ones(size(C2F2Test, 1), 1)];
            elseif foldNum == 3
                TrainData = [C1F3Train ; C2F3Train];
                TestData = [C1F3Test ; C2F3Test];
                Labels = [ones(size(C1F3Train, 1), 1) ; -ones(size(C2F3Train, 1), 1)];
                TestLabels = [ones(size(C1F3Test, 1), 1) ; -ones(size(C2F3Test, 1), 1)];
            end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Here we start some Kernel calculations
            % N is the size of the train data
            % Dim will be 2 because the matrices are the number of samples
            % by 2.
            [N,Dim] = size(TrainData);
            % Inisitiate the Hessian Matrix and fill it with all zeros the
            % same size as the training data being used
            H = zeros(length(TrainData));
            % Set a variable to the length of the training data
            len = length(TrainData);
            % Populate an [i x j] matrix called H with the label at
            % position i times the label at position j times k. i and j
            % iterate through for loops until they reach the length of the
            % training data
            if KernelIndex == 1
                HMatrixPoly = zeros(len);
            end
            
            if KernelIndex == 2
                HMatrixRad = zeros(len);
            end
            
            for i = 1:len
                for j = 1:len
                    if KernelIndex == 1
                        % Polynomial Kernel Function
                        % K(Xi, Xj) = ((Xi)^T * Xi +1)^2
                        K = (TrainData(j,:) * TrainData(i,:)' + 1)^2;
                        % H(i,j) = Yi * Yj ((Xi)^T * Xj + 1)^2
                        H(i,j) = Labels(i)*Labels(j)*K;
                        HMatrixPoly(i,j) = H(i,j);
                    elseif KernelIndex == 2
                        % Radial Basis Functions Kernel
                        % K(X, Z) = exp((- (abs(X - Z))^2) / (sigma)^2 
                        K = exp(-((norm(Data(i,:) - Data(j,:))^2)/100));
                        % H(i,j) = Yi * Yj ((Xi)^T * Xj + 1)^2
                        H(i,j) = Labels(i)*Labels(j)*K;
                        HMatrixRad(i,j) = H(i,j);                     
                    end                   
               end
            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Now we need to have an if statement that will keep the c
            % value updated to the one we want to use
            if (z == 1)
                Cval = 10;
            elseif (z == 2)
                Cval = 100;
            elseif (z == 3)
                Cval = 1000;
            end

            ChangeToMin = -ones(N,1);
            Aeq = zeros(N,N);
            Aeq(1,:) = Labels;
            beq = zeros(N,1);
            H = ((TrainData * TrainData') * (Labels' * Labels));
            [alpha] = quadprog(H + eye(N) * 0.0001, ChangeToMin, [], [], Aeq, beq, zeros(1,N), Cval * ones(1,N));
            supportVectors = find(alpha > .00000000001);
            supportX = alpha(supportVectors);
            supportData = Data(supportVectors,:);
            supportLabels = Labels(supportVectors);
            supportLength = length(supportLabels);

            %Now, solve for b
            %Create a set of b's and average them
            Bset = [];
            for i=1 : supportLength
                Bval = 0;
                for j = 1 : supportLength
                    if KernelIndex == 1
                        % Polynomial Kernel Function
                        % K(Xi, Xj) = ((Xi)^T * Xi +1)^2
                        K = (supportData(i,:)*supportData(j,:)' + 1)^2;
                        Bval = Bval + (supportX(j) * supportLabels(j) * K );
                        BvalPoly = Bval;
                    elseif KernelIndex == 2
                        % Radial Basis Functions Kernel
                        % K(X, Z) = exp((- (abs(X - Z))^2) / (sigma)^2 
                        K = exp(-((norm(supportData(i,:) - supportData(j,:))^2)/100));
                        Bval = Bval + (supportX(j) * supportLabels(j) * K );
                        BvalRad = Bval;                    
                    end  
                end
                Bval = supportLabels(i) * Bval;
                Bval = (1 - Bval)/supportLabels(i);
                Bset = [Bset Bval];
            end
            b = mean(Bset);

            Res = zeros(1,size(TestData,1));
            for i = 1 : size(TestData,1)
                sumVal = 0;
                for j = 1 : supportLength
                    if KernelIndex == 1
                        % Polynomial Kernel Function
                        % K(Xi, Xj) = ((Xi)^T * Xi +1)^2
                        K = (supportData(j,:)*TestData(i,:)' + 1)^2;
                        sumVal = sumVal + supportX(j)*supportLabels(j)*K;
                        sumValPoly = sumVal;
                    elseif KernelIndex == 2
                        % Radial Basis Functions Kernel
                        % K(X, Z) = exp((- (abs(X - Z))^2) / (sigma)^2 
                        K = exp(-((norm(TestData(i,:) - supportData(j,:))^2)/100));
                        sumVal = sumVal + supportX(j)*supportLabels(j)*K;
                        sumValRad = sumVal;                    
                    end
                    
                end
                Res(i) = sumVal + b;
            end
            
            CM = zeros(2, 2);
            
            for i = 1 : size(TestData, 1)
                CorrectClass = TestLabels(i, 1);
                Classifier = Res(1, i);
    
                if (CorrectClass > 0 && Classifier > 0)
                    CM(1,1) = CM(1,1) + 1;
                end
    
                if (CorrectClass > 0 && Classifier < 0)
                    CM(2,1) = CM(2,1) + 1;
                end
    
                if (CorrectClass < 0 && Classifier < 0)
                    CM(2,2) = CM(2,2) + 1;
                end
    
                if (CorrectClass < 0 && Classifier > 0)
                    CM(1,2) = CM(1,2) + 1;
                end
            end
            
            if KernelIndex == 1
                if (z == 1 && foldNum == 1)
                    CM(:, 1) =  (CM(:, 1)./size(C1F1Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F1Test, 1) * 100);
                    PolyCM_Cval10F1 = CM;
                    plotNum = 2;
                    figure(plotNum);
                    title('CVal 10 Fold 1 Polynomial');
                    hold on;
                    for i = 1 : size(C1F1Test, 1)
                        plot(C1F1Test(i,1), C1F1Test(i,2), 'bx');
                        %plot(supportData(i,1), supportData(i,2), 'ro');
                    end
                    for i = 1 : size(C2F1Test, 1)
                        plot(C2F1Test(i,1), C2F1Test(i,2), 'ro');
                    end
                    hold off;                    
                elseif (z == 1 && foldNum == 2)
                    CM(:, 1) =  (CM(:, 1)./size(C1F2Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F2Test, 1) * 100);
                    PolyCM_Cval10F2 = CM;
                elseif (z == 1 && foldNum == 3)
                    CM(:, 1) =  (CM(:, 1)./size(C1F3Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F3Test, 1) * 100);
                    PolyCM_Cval10F3 = CM;
                elseif (z == 2 && foldNum == 1)
                    CM(:, 1) =  (CM(:, 1)./size(C1F1Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F1Test, 1) * 100);
                    PolyCM_Cval100F1 = CM;    
                elseif (z == 2 && foldNum == 2)
                    CM(:, 1) =  (CM(:, 1)./size(C1F2Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F2Test, 1) * 100);
                    PolyCM_Cval100F2 = CM;
                elseif (z == 2 && foldNum == 3)
                    CM(:, 1) =  (CM(:, 1)./size(C1F3Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F3Test, 1) * 100);
                    PolyCM_Cval100F3 = CM;
                elseif (z == 3 && foldNum == 1)
                    CM(:, 1) =  (CM(:, 1)./size(C1F1Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F1Test, 1) * 100);
                    PolyCM_Cval1000F1 = CM;
                elseif (z == 3 && foldNum == 2)
                    CM(:, 1) =  (CM(:, 1)./size(C1F2Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F2Test, 1) * 100);
                    PolyCM_Cval1000F2 = CM;
                elseif (z == 3 && foldNum == 3)
                    CM(:, 1) =  (CM(:, 1)./size(C1F3Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F3Test, 1) * 100);
                    PolyCM_Cval1000F3 = CM;
                end
            end
            
            if KernelIndex == 2
                if (z == 1 && foldNum == 1)
                    CM(:, 1) =  (CM(:, 1)./size(C1F1Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F1Test, 1) * 100);
                    RadCM_Cval10F1 = CM;
                elseif (z == 1 && foldNum == 2)
                    CM(:, 1) =  (CM(:, 1)./size(C1F2Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F2Test, 1) * 100);
                    RadCM_Cval10F2 = CM;
                elseif (z == 1 && foldNum == 3)
                    CM(:, 1) =  (CM(:, 1)./size(C1F3Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F3Test, 1) * 100);
                    RadCM_Cval10F3 = CM;
                elseif (z == 2 && foldNum == 1)
                    CM(:, 1) =  (CM(:, 1)./size(C1F1Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F1Test, 1) * 100);
                    RadCM_Cval100F1 = CM;    
                elseif (z == 2 && foldNum == 2)
                    CM(:, 1) =  (CM(:, 1)./size(C1F2Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F2Test, 1) * 100);
                    RadCM_Cval100F2 = CM;
                elseif (z == 2 && foldNum == 3)
                    CM(:, 1) =  (CM(:, 1)./size(C1F3Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F3Test, 1) * 100);
                    RadCM_Cval100F3 = CM;
                elseif (z == 3 && foldNum == 1)
                    CM(:, 1) =  (CM(:, 1)./size(C1F1Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F1Test, 1) * 100);
                    RadCM_Cval1000F1 = CM;
                elseif (z == 3 && foldNum == 2)
                    CM(:, 1) =  (CM(:, 1)./size(C1F2Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F2Test, 1) * 100);
                    RadCM_Cval1000F2 = CM;
                elseif (z == 3 && foldNum == 3)
                    CM(:, 1) =  (CM(:, 1)./size(C1F3Test, 1) * 100);
                    CM(:, 2) =  (CM(:, 2)./size(C2F3Test, 1) * 100);
                    RadCM_Cval1000F3 = CM;
                end
            end
        end
    end
end

