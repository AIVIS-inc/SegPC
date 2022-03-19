%%
clc, clear, close all;

img_path = '../dataset/'; %input image path
img_path_out = '../merged_post/'; %output image path
imagefiles = dir('../dataset/*.bmp');     
imagefiles_infer = dir('../merged/*.png');

%% file name of inference
nfiles_infer = length(imagefiles_infer);    % Number of files found
infer_split = zeros(length(imagefiles_infer), 2);

% The infer_split is used as the index
for jj = 1:nfiles_infer
    infer_name = imagefiles_infer(jj).name;
    infer_split_cell = split(infer_name,["_","."]);       % Image name split: ex: 9913_6.png --> 9913, 6, png
    infer_split(jj, 1) = str2double(infer_split_cell{1}); % infer_split_cell{1} = 9913
    infer_split(jj, 2) = str2double(infer_split_cell{2}); % infer_split_cell{2} = 6
end
indexMat = infer_split(:,1);
indexMat_ImgSub = infer_split(:,2);
%%
nfiles = length(imagefiles);    % Number of files found
for ii=1:nfiles
    currentfilename = imagefiles(ii).name;                  % ex: currentfilename=101.bmp 
    img_Name = extractBefore(currentfilename, ".");         % ex: img_Name=101 
    indexMatched = find(indexMat == str2double(img_Name));  % indexMat is obtained from line 20
    
    
    % The image obtained via post processing is named according to the "img_subname_sort(end)"
    img_subname = indexMat_ImgSub(indexMatched);
    img_subname_sort = sort(img_subname);    
    bias_subname = img_subname_sort(end)+1;
    
    % Read and resize image
    img = sprintf('%s%s', img_path, currentfilename);
    he_in =imread(img);
    he = imresize(he_in, [1080, 1440]);
    
    
    % Change the segmentation result of nucleus to binary [nuSet]
    [nuSet, lowerBound, upperBound] = StaticComputation(indexMatched, imagefiles_infer, he);
    
    % Color space transformation
    img_hsv = rgb2hsv(he);
    saturation = img_hsv(:,:,2);
    saturation2 = imguidedfilter(saturation, 'NeighborhoodSize', [30, 30], 'DegreeOfSmoothing', 100);
    saturation2(saturation2 < 0.2) = 0;
    lab_he = rgb2lab(he);
%     ab = zeros(size(he));
%     ab(:,:,1) = saturation;
    ab = lab_he(:,:,2:3);
    ab = im2single(he);
    nColors = 3;
    
    % Matlab inner-function(imsegkmeans): --K-means clustering
    %  L = imsegkmeans(I,k) segments the input image I into k clusters by
    %  performing k-means clustering and returns the segmented labeled output
    pixel_labels = imsegkmeans(ab, nColors, 'NumAttempts', 2);

    % choose label based on the static information
    % The BW with the largest label is choosen
    BW = labelChose(nuSet, pixel_labels, he);
    
    % Label the logic image BW. 
    [L, n] = bwlabel(BW); 
    if(n < 1)
        continue;
    end
    
    
    % Filter out some labels in L by thresholding the number of pixels. The
    % threshold is obtained by line:42. Only the label in the specific
    % range is used to produce final result.
    nbins = 1:n;
    data_hist = histcounts(L(:), nbins);
    idx_Valid_lower = find(data_hist > lowerBound);
    idx_Valid_upper = find(data_hist < upperBound);
    [val,pos]=intersect(idx_Valid_lower, idx_Valid_upper);
    
    

    % idx represents valid idx    
    label = 1;
    LabelAdd = zeros(size(L));
    for i = 1:size(val,2)
        L_zero = zeros(size(L));
        valid_i = val(i);
        
        %Extract valid label
        idx_valid = find(L == valid_i); % L is obtained from line 79
        L_zero(idx_valid) = 1; 
        
        % Check whether the classified label is already exist.
        iouMeasure = iou(L_zero, nuSet); 
        if(iouMeasure == 0)
            % Construct a convex hull.
            bw2 = bwconvhull(L_zero);
            
            % Obtain cytoplasm region via dialization and saturation value.  
            se = strel('disk',30);
            dilatedI = imdilate(uint8(bw2), se);
            cy = dilatedI - uint8(bw2);
            cy_candidate = saturation2 .*  double(cy);
            index_cy=(cy_candidate~=0);
            
            L_zero = bw2 * 40;
            L_zero(index_cy) = 20;
            % Image name 
            imgAddName = sprintf('%s%s_%d.png', img_path_out, img_Name, bias_subname);
            imwrite(uint8(L_zero), imgAddName);
            bias_subname = bias_subname+1;
        end
    end

end

