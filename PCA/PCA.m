function [arrange_eigenvalue, k_] = skeleton_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];    % image size (rows,columns)
    X=load_faces();% Load the AT&T Face data set using load_faces()
    %%%%% TODO

    %% Compute mean face and the covariance matrix of faces
    % compute X_tilde
    X_mean = mean(X, 1);
    X_tilde = X - X_mean;
    %%%%% TODO
    X_cov = (1/400)*X_tilde.'*X_tilde;% Compute covariance matrix using X_tilde
    %%%%% TODO
    
    %% Compute the eigenvalue decomposition of the covariance matrix
    [eigenvector,eigenvalue] = eig(X_cov);
    [eigen_value, D] = sort(diag(eigenvalue),'descend');
    %%%%% TODO
    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    
    %value = eigenvalue(D,D);
    eigen_vector = eigenvector(:,D);
    
    %%%%% TODO
    
    %% Compute the principal components: Y
    test_img_idx = 43;
    test_img = X(test_img_idx,:);    
    % Compute eigenface coefficients
    %%%% TODO
    y = zeros(450,10304);
    % Compute the principal components: Y
    for m = 1:450
        y(m,:) = eigen_vector(:,m).'*(test_img-X_mean).'*eigen_vector(:,m);
    end
    %%%%% TODO

%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    imshow(uint8(reshape(X(120,:), img_size))) 
    title('#120 picture')
    %%%%% TODO
    subplot(1,2,2)
    imshow(uint8(reshape(X_mean, img_size)))
    title('mean picture')
    % Visualize the mean face image
    %%%%% TODO
    
%% Q5.2 b) Analysing computed eigenvalues
    
    warning('off')
    load('eigen_value');
    % Report the top 5 eigenvalues
    lambda_top5=zeros(5,1);
    for i = 1:5
        lambda_top5(i,1) = eigen_value(i,1); %%%%% TODO
    end
    % Plot the eigenvalues in from largest to smallest
    %k = 1:d;
    k = 1:1:450; 
    figure(2)
    
    sgtitle('Eigenvalues from largest to smallest')
    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    plot(k,eigen_value(1:450,1),'k');
    xlabel('k value')
    ylabel('eigen_value  value')
    title(' eigen_value  value ')
    %%%%% TODO
    p = zeros(450,1);
    Lambda_d = sum(eigen_value);
    for j = 1:450
        Lambda_k = sum(eigen_value(1:j,1));
        p(j,1) = round(Lambda_k/Lambda_d,2);
    end
    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    subplot(1,2,2)
    plot(k,p,'k');
    xlabel('k value')
    ylabel('ρk  value')
    title(' the values of ρk ')
    %%%%% TODO
    
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    small_k = zeros(2,5);
    for a = 1:5
        num = sum(sum(p<ef(a)));
        small_k(2,a) = num +1;
        small_k(1,a) = ef(a);
    end
    %%%%% TODO (Hint: ismember())
    % k_ = ?; %%%%% TODO
    
    
%% Q5.2 c) Approximating an image using eigen faces
    
    K = [0, 1, 2, small_k(2,:),400,450];
    figure(3)
    for n = 1:9
        X_hat = X_mean+sum(y(1:K(n),:));
        subplot(3,3,n);
        imshow(uint8(reshape(X_hat, img_size)));
        c = K(n);
        title(['k=',num2str(c)]);
    end
    subplot(3,3,2);
    X_hat = X_mean+y(1,:);
    imshow(uint8(reshape(X_hat, img_size)));
    title('k=1');
    %%%% TODO 
    sgtitle('Approximating original image by adding eigen faces')

%% Q5.2 d) Principal components capture different image characteristics
    %% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of quantile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,~] = size(X);
    X = X.';
    
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    X_mean = mean(X, 2);
    X_tilde = X - X_mean;
    %%%%% TODO
    % Compute covariance using X_tilde
    X_cov = (1/400)*(X_tilde*X_tilde.');
    %%%%% TODO
    
    %% Compute the eigenvalue decomposition
    [eigenvector,eigenvalue] = eig(X_cov);
    %%%%% TODO
    
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    [eigen_value, D] = sort(diag(eigenvalue),'descend');
    eigen_vector = eigenvector(:,D);
    %%%%% TODO
    
   %% Compute principal components
    %%%%% TODO     *eigen_vector(:,m);
    %y_hat=eigen_vector.'*X_tilde;
      
    %% Computing the first 2 pricipal components
    %%%%% TODO
%     y_hat1 = zeros(1,256);
%     y_hat2 = zeros(1,256);
    y_hat1 = eigen_vector(:,1).'*X_tilde;
    y_hat2 = eigen_vector(:,2).'*X_tilde;
    %y_hat2 = y_hat(2,:).';
    y_point = [y_hat1; y_hat2];
    % finding quantile points
    quantile_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - quantile_points())
    
    % Finding the cartesian product of quantile points to find grid corners
    %%%%% TODO
    percentile1 = percentile_values((y_hat1).', quantile_vals);
    percentile2 = percentile_values((y_hat2).', quantile_vals);
    
    [Xaxis,Yaxis] = meshgrid(percentile1,percentile2);
    Cartesian = [Xaxis(:),Yaxis(:)];
    %scatter(Cartesian(:,1),Cartesian(:,2),'ro');
    
    %% Find images whose PCA coordinates are closest to the grid coordinates 

    dist_min= zeros(1,25);
    dist_index=zeros(1,25);
    for c = 1:25
        dist_2= zeros(1,658);
        for p = 1:658
            dist_2(1,p)=(Cartesian(c,1)-y_hat1(1,p)).^2+(Cartesian(c,2)-y_hat2(1,p)).^2;
        end
        [dist_min(1,c), dist_index(c)]=min(dist_2);
    end
    red_point = zeros(2,25);
    for a = 1:25
        q = dist_index(a);
        red_point(1,a) = y_hat1(q);
        red_point(2,a) = y_hat2(q);
    end
    
    %[dist_min,dist_index]=min(Car_point,[],2);
    %%%%% TODO

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')
    
    % Visualize the 100th image
    subplot(1,2,1)
    %%%%% TODO
    imshow(reshape(X(:,120), img_size)); 
    title('#120 dataset');
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(reshape(mean(X,2), img_size));
    title('mean dataset');
    
    %% Image projections onto principal components and their corresponding features
    
    figure(5) 
    hold on
    grid on  
    scatter(y_hat1,y_hat2,'bo');
    scatter(red_point(1,:), red_point(2,:), 'o', 'MarkerFaceColor', 'r');
    xticks(percentile1);
    yticks(percentile2);
    title('first two principal components of all images');
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the quantile points and highlight the image points that are closest to the 
    % quantile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)

    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to quantile grid corners')
    hold off
    
    figure(6)
   
    for g = 1:25
         v=dist_index(g);
        
        subplot(5,5,g)
        imshow(reshape(X(:,v), img_size)); 
    end
    
%   sgtitle('Images closest to quantile grid corners')
%   hold on
    % Plot the images whose PCA coordinates are closest to the quantile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
    
    hold off    
end
function [percentile_values] = percentile_values(v, percentiles)
    % assumes v as a column vector
    % percentiles is an array of percentiles 
    % percentile_values is an array of percentile-values corresponding
    % to percentiles 
    [n, ~] = size(v);
    [sorted_v,~] = sort(v, 'ascend');
    percentile_indices = ceil(n*percentiles/100);
    percentile_values = sorted_v(percentile_indices,:);
end
