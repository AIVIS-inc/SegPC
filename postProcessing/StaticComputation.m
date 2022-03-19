function [nuSet, lowerBound, upperBound] = StaticComputation(indexMatched, imagefiles_infer, img_Ori)
% Read matched image from imagefiles_infer based on the indexMatched
cuntNuPixel = zeros(length(indexMatched), 1);

% For each nucleus, count the number of the pixel of it. 
for i = 1:length(indexMatched)
    imgName = sprintf('%s/%s', imagefiles_infer(indexMatched(i)).folder, imagefiles_infer(indexMatched(i)).name);
    img = imread(imgName);
    if(i==1)
        nuSet = zeros(size(img));
    end
    nu_i = find(img == 40);
%     cy_i = find(img == 20);
    cuntNuPixel(i,1) = nnz(nu_i);
    % Change the segmentation result to binary
    nuSet(nu_i) = 1; 
end

meanCunt = mean(cuntNuPixel);

minCunt = min(cuntNuPixel);
lowerBound = minCunt*0.1;

MaxCunt = max(cuntNuPixel);
upperBound = MaxCunt + meanCunt*0.8;


%% For cy
% meancy = mean(sat_cyPixel_mean);
% mincy = min(sat_cyPixel_min);
% maxcy = max(sat_cyPixel_max);
% 
% lowerBound_cy = mincy*0.1;
% upperBound_cy = maxcy + meancy*0.1;

