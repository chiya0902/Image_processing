%face recognition using transfer learning

%train model:
clc
clear all
g = alexnet;
layers = g.Layers;
layers(23) = fullyConnectedLayer(2);
layers(25) = classificationLayer;
allImages = imageDatastore('C:\Users\Abhishree\Pictures\wo','IncludeSubFolders',true,'LabelSource','foldernames');
opts = trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
myNet = trainNetwork(allImages,layers,opts);
save myNet;

%testing model:
clc;close;clear
c = webcam;
load myNet;
faceDetector = vision.CascadeObjectDetector;
while true
    e = c.snapshot;
    bboxes = step(faceDetector,e);
    if (sum(sum(bboxes))~=0)
        es = imcrop(e,bboxes(1,:));
        es = imresize(es,[227 227]);
        label = classify(myNet,es);
        image(e);
        title(char(label));
        drawnow;
    else
        image(e)
        title('no face detected');
    end
end