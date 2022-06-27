function [trainedNet, info] = trainNetwork(varargin)
% trainNetwork   Train a neural network
%
%   trainedNet = trainNetwork(imds, layers, options) trains and returns a
%   network trainedNet for a classification problem. imds is an
%   ImageDatastore with categorical labels, layers is an array of network
%   layers or a LayerGraph, and options is a set of training options.
%
%   trainedNet = trainNetwork(ds, layers, options) trains and returns a
%   network trainedNet using the datastore ds. For single-input networks,
%   the datastore read function must return a two-column table or
%   two-column cell array, where the first column specifies the inputs to
%   the network and the second column specifies the expected responses. For
%   networks with multiple inputs, the datastore read function must return
%   a cell array with N+1 columns, where N is the number of inputs. The
%   first N columns correspond to the N inputs and the final column
%   corresponds to the responses.
%
%   trainedNet = trainNetwork(X, Y, layers, options) trains and returns a
%   network, trainedNet. The format of X depends on the input layer.
%      - For an image input layer, X is a numeric array of images arranged
%        so that the first three dimensions are the width, height and
%        channels, and the last dimension indexes the individual images.
%      - For a 3-D image input layer, X is a numeric array of 3-D images
%        with the dimensions width, height, depth, channels, and the last
%        dimension indexes the individual observations.
%      - For a feature input layer, X is a numeric array arranged so that
%        the first dimension indexes the individual observations, and the
%        second dimension indexes the features of the data set.
%   The format of Y depends on the type of task.
%      - For a classification task, Y specifies the labels for the data/images
%        as a categorical vector. 
%      - For a regression task, Y contains the responses arranged
%        as a matrix of size number of observations by number of responses.
%        When dealing with images, Y can also be specified as a four
%        dimensional numeric array, where the last dimension corresponds to 
%        the number of observations.
%
%   trainedNet = trainNetwork(sequences, Y, layers, options) trains an LSTM
%   network for classification and regression problems for sequence or
%   time-series data. layers must define a network with a sequence input
%   layer. sequences must be one of the following:
%      - A cell array of C-by-S matrices, where C is the number of features
%        and S is the number of time steps.
%      - A cell array of H-by-W-by-C-by-S arrays, where H-by-W-by-C is the
%        2-D image size and S is the number of time steps.
%      - A cell array of H-by-W-by-D-by-C-by-S arrays, where
%        H-by-W-by-D-by-C is the 3-D image size and S is the number of time
%        steps.
%   Y must be one of the following:
%      - For sequence-to-label classification, a categorical vector.
%      - For sequence-to-sequence classification, a cell array of
%        categorical sequences.
%      - For sequence-to-one regression, a matrix of targets.
%      - For sequence-to-sequence regression, a cell array of C-by-S
%        matrices.
%   For sequence-to-sequence problems, the number of time steps of the
%   sequences in Y must be identical to the corresponding predictor
%   sequences. For sequence-to-sequence problems with one observation, the
%   input sequence can be a numeric array, and Y must be a categorical
%   sequence of labels or a numeric array of responses.
%
%   trainedNet = trainNetwork(tbl, layers, options) trains and returns a
%   network, trainedNet.
%      - For networks with an image input layer, tbl is a table
%        containing predictors in the first column as a cell array
%        of image paths or images. Responses must be in the second column
%        as categorical labels for the images. In a regression problem,
%        responses must be in the second column as either vectors or cell
%        arrays containing 3-D arrays or in multiple columns as scalars.
%      - For networks with a sequence input layer, tbl is a table containing
%        a cell array of MAT file paths of predictors in the first column.
%        For a sequence-to-label classification problem, the second column
%        must be a categorical vector of labels. For a sequence-to-one
%        regression problem, the second column must be a numeric array of
%        responses or in multiple columns as scalars. For a sequence-to-sequence
%        classification problem, the second column must be a cell array of
%        MAT file paths with a categorical response sequence. For a
%        sequence-to-sequence regression problem, the second column must
%        be a cell array of MAT file paths with a numeric response sequence.
%        Support for tables and networks with a sequence input layer will
%        be removed in a future release. For out-of-memory data, use a
%        datastore instead.
%      - For networks with a feature input layer, tbl is a table
%        containing predictors in multiple columns as scalars and responses
%        as a column of categorical labels. For regression tasks, responses
%        must be in multiple columns as scalars or in a single column as a
%        numeric vector.
%
%   trainedNet = trainNetwork(tbl, responseNames, layers, options) trains
%   and returns a network, trainedNet. responseNames is a character
%   vector, a string array, or a cell array of character vectors specifying
%   the names of the variables in tbl that contain the responses.
%
%   [trainedNet, info] = trainNetwork(...) trains and returns a network,
%   trainedNet. info contains information on training progress.
%
%   Example 1:
%       % Train a convolutional neural network on some synthetic images
%       % of handwritten digits. Then run the trained network on a test
%       % set, and calculate the accuracy.
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [ ...
%           imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 2:
%       % Train a long short-term memory network to classify speakers of a
%       % spoken vowel sounds on preprocessed speech data. Then make
%       % predictions using a test set, and calculate the accuracy.
%
%       [XTrain, YTrain] = japaneseVowelsTrainData;
%
%       layers = [ ...
%           sequenceInputLayer(12)
%           lstmLayer(100, 'OutputMode', 'last')
%           fullyConnectedLayer(9)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('adam', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = japaneseVowelsTestData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 3:
%       % Train a network on synthetic digit data, and measure its
%       % accuracy:
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [
%           imageInputLayer([28 28 1], 'Name', 'input')
%           convolution2dLayer(5, 20, 'Name', 'conv_1')
%           reluLayer('Name', 'relu_1')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
%           reluLayer('Name', 'relu_2')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
%           reluLayer('Name', 'relu_3')
%           additionLayer(2,'Name', 'add')
%           fullyConnectedLayer(10, 'Name', 'fc')
%           softmaxLayer('Name', 'softmax')
%           classificationLayer('Name', 'classoutput')];
%
%       lgraph = layerGraph(layers);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
%
%       plot(lgraph);
%
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       [net,info] = trainNetwork(XTrain, YTrain, lgraph, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   See also nnet.cnn.layer, trainingOptions, SeriesNetwork, DAGNetwork, LayerGraph.

%   Copyright 2015-2020 The MathWorks, Inc.

narginchk(3,4);

try
    factory = nnet.internal.cnn.trainNetwork.DLTComponentFactory();
    [trainedNet, info] = nnet.internal.cnn.trainNetwork.doTrainNetwork(factory,varargin{:});
catch e
    iThrowCNNException( e );
end

end

function iThrowCNNException( exception )
% Wrap exception in a CNNException, which reports the error in a custom way
err = nnet.internal.cnn.util.CNNException.hBuildCustomError( exception );
throwAsCaller(err);
end
