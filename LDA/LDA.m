%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2021
% HW 4
% <shuyi fan    shuyifan@bu.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i) 

lambda1 = 1;
lambda2 = 0.25;
theta = 0*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussias which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure;subplot(2,2,1);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['lambda1 = 1; lambda2 = 0.25;\theta = ',num2str(0),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
% ...
lambda1 = 1;
lambda2 = 0.25;
theta = 2*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);
subplot(2,2,3);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['lambda1 = 1; lambda2 = 0.25;\theta = ',num2str(2),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

lambda1 = 0.25;
lambda2 = 1;
theta = 1*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);
subplot(2,2,4);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['lambda1 = 0.25; lambda2 = 1;\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

lambda1 = 1;
lambda2 = 0.25;
theta = 1*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);
X1 = X(:, Y==1);
X2 = X(:, Y==2);
subplot(2,2,2);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['lambda1 = 1; lambda2 = 0.25;\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal_power;
    noise_power_array(i) = noise_power;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%signal power

max_signal_power=max(signal_power_array);
xaxis_signal_max=find(signal_power_array==max_signal_power);
figure;
plot(phi_array,signal_power_array)
    hold on;
plot(phi_array(xaxis_signal_max),max_signal_power,'ro');
    hold on;
text(phi_array(xaxis_signal_max),max_signal_power,['X=',num2str(phi_array(xaxis_signal_max)),char(10),'Y=',num2str(max_signal_power)]);
xlabel('phi value')
ylabel('signal power value')
title('signal power versus phi & its maximum')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%noise power
% min_noise_power=min(noise_power_array);
% xaxis_noise_min=find(noise_power_array==min_noise_power);
% plot(phi_array,noise_power_array)
%     hold on;
% plot(phi_array(xaxis_noise_power),min_noise_power,'ro');
% %     hold on;
% text(phi_array(xaxis_noise_power),min_noise_power,['X=',num2str(phi_array(xaxis_noise_power)),char(10),'Y=',num2str(min_noise_power)]);
% xlabel('phi value')
% ylabel('noise power value')
% title('noise power versus phi & its minimum')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %snr
% max_snr=max(snr_array);
% xaxis_snr=find(snr_array==max_snr);
% plot(phi_array,snr_array)
%     hold on;
% plot(phi_array(xaxis_snr),max_snr,'ro');
%     hold on;
% text(phi_array(xaxis_snr),max_snr,['X=',num2str(phi_array(xaxis_snr)),char(10),'Y=',num2str(max_snr)]);
% xlabel('phi value')
% ylabel('snr value')
% title('snr versus phi & its maximum')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
phi_array = [0  pi/6  pi/3];
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), true);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal_power;
    noise_power_array(i) = noise_power;
    snr_array(i) = snr;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);
% 
% % See below for the LDA function which you need to complete.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Insert code to create a scatter plot and overlay the LDA vector and the 
% % difference between the class means. Use can use Matlab's quiver function 
% % to do this.
% % ...
figure;
hold on;
mean_X1=mean(X1,2);
mean_X2=mean(X2,2);
mean_vector=mean_X2-mean_X1;
hold on;
plot(X1(1,:),X1(2,:),'k.');
hold on;
plot(X2(1,:),X2(2,:),'b.');
hold on;
quiver(mean_X1(1),mean_X1(2),w_LDA(1,1)-mean_X1(1),w_LDA(2,1)-mean_X1(2));
hold on;
quiver(mean_X1(1),mean_X1(2),mean_vector(1,1),mean_vector(2,1));
hold on;
legend('w_LDA_arrows','mean_arrow');
xlabel('X axis')
ylabel('Y axis')
title('vectors represented as arrows')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n=150;
w_LDA = LDA(X,Y);
% Create CCR vs b plot

X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
max_ccr=max(ccr_array);
xaxis_b=find(ccr_array==max_ccr);
figure;
plot(b_array,ccr_array,'k');
hold on;
plot(b_array(xaxis_b),max_ccr,'ro');
    hold on;
xlabel('b value')
ylabel('CCR value')
title('CCR versus b & its maximum')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%%%%%%%%%%%%%%%%%%%%%%
%Insert your code here
%calculate the 2 × 2 covariance matrix
U1=[cos(theta) sin(theta)];
U2=[sin(theta) -cos(theta)];
U=[U1; U2];      %eigenvectors
Lambda= [lambda1 0; 0 lambda2];
C=(U*Lambda)/ U;
%generate Class1 data
Class1=mvnrnd(mu1, C ,n1);
%X1=Class1.';
%generate Class1 data
Class2=mvnrnd(mu2, C ,n2);
%X2=Class2.';
X=[Class1; Class2].';
Y1=ones(1,50);
Y2=2*ones(1,100);
Y=[Y1 Y2].';
% See below for function two_2D_Gaussias which you need to complete.
%%%%%%%%%%%%%%%%%%%%%%
end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ...
n1 = 50;
n2 = 100;
n=n1+n2;
X1 = X(:, Y==1);
X2 = X(:, Y==2);
% (i)squared_euclidean
% signal  
signal1= [cos(phi) sin(phi)]*mean(X1,2);
signal2= [cos(phi) sin(phi)]*mean(X2,2);
signal = (signal2-signal1)*(signal2-signal1).';
% 
% (ii)noise
noise1 = [cos(phi) sin(phi)]*(X1-mean(X1,2))*(X1-mean(X1,2)).'*[cos(phi) sin(phi)].'/(n1);
noise2 = [cos(phi) sin(phi)]*(X2-mean(X2,2))*(X2-mean(X2,2)).'*[cos(phi) sin(phi)].'/(n2);
noise= (n1/n)*noise1+(n2/n)*noise2;

% (iii)SNR
snr = signal/noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity([cos(phi) sin(phi)]*X1);
    plot(pdf1,z1)
    hold on;
    [pdf2,z2] = ksdensity([cos(phi) sin(phi)]*X2);
    plot(pdf2,z2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title('Estimated class density estimates of data projected along \phi = 2 \times \pi/6. Ground-truth \phi = \pi/6')
end


end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% ...
n1 = 50;
n2 = 100;
n=n1+n2;
X1 = X(:, Y==1);
X2 = X(:, Y==2);
S1=(1/n1)*(X1-mean(X1,2))*(X1-mean(X1,2)).';
S2=(1/n2)*(X2-mean(X2,2))*(X2-mean(X2,2)).';
S=(n1/n)*S1+(n2/n)*S2;
S_Inverse=inv(S);
w_LDA=S_Inverse*(mean(X2,2)-mean(X1,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
w=w_LDA;
H=w.'*X+b;    % 1X2  2X150      
H1=H(1,[1:50]);
H2=H(1,[51:150]);
ccr1=sum(sum(H1<=0));
ccr2=sum(sum(H2>0));
ccr=(ccr1+ccr2)/150;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end