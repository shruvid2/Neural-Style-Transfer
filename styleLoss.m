function loss = styleLoss(transferStyleFeatures,styleFeatures,styleWeights)

    loss = 0;
    for i=1:numel(styleFeatures)
        
        tsf = transferStyleFeatures{1,i};
        sf = styleFeatures{1,i};    
        [h,w,c] = size(sf);
        
        gramStyle = computeGramMatrix(sf);
        gramTransfer = computeGramMatrix(tsf);
        sLoss = mean((gramTransfer - gramStyle).^2,'all') / ((h*w*c)^2);
        
        loss = loss + (styleWeights(i)*sLoss);
    end
end