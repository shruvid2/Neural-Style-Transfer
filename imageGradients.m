function [gradients,losses] = imageGradients(dlnet,dlTransfer,contentFeatures,styleFeatures,params)
 
    % Initialize transfer image feature containers. 
    numContentFeatureLayers = numel(params.contentFeatureLayerNames);
    numStyleFeatureLayers = numel(params.styleFeatureLayerNames);
 
    transferContentFeatures = cell(1,numContentFeatureLayers);
    transferStyleFeatures = cell(1,numStyleFeatureLayers);
 
    % Extract content features of transfer image.
    [transferContentFeatures{:}] = forward(dlnet,dlTransfer,'Outputs',params.contentFeatureLayerNames);
     
    % Extract style features of transfer image.
    [transferStyleFeatures{:}] = forward(dlnet,dlTransfer,'Outputs',params.styleFeatureLayerNames);
 
    % Compute content loss. 
    cLoss = contentLoss(transferContentFeatures,contentFeatures,params.contentFeatureLayerWeights);
 
    % Compute style loss. 
    sLoss = styleLoss(transferStyleFeatures,styleFeatures,params.styleFeatureLayerWeights);
 
    % Compute final loss as weighted combination of content and style loss. 
    loss = (params.alpha * cLoss) + (params.beta * sLoss);
 
    % Calculate gradient with respect to transfer image.
    gradients = dlgradient(loss,dlTransfer);
    
    % Extract various losses. 
    losses.totalLoss = gather(extractdata(loss));
    losses.contentLoss = gather(extractdata(cLoss));
    losses.styleLoss = gather(extractdata(sLoss));
 
end