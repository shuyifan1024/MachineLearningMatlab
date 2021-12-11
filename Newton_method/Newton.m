%% Newton's method
clear all; clc;
load('data.mat');
%% Input Data features and labels
X_total = dataprocessed{:,1:9};  % features of all data
% X_total(:,1) = X_total(:,1)/40;
% X_total(:,3) = X_total(:,3)/15;
% X_total(:,7) = X_total(:,7)/4000;
% X_total(:,8) = X_total(:,8)/1800;
% X_total(:,9) = X_total(:,9)/50;
[n,d] = size(X_total);  % measure the size of data
X_data_train = X_total(1:20000,:); % train data
X_data_test = X_total(20001:29170,:); % test data

Y_total = dataprocessed{:,10}; % labels of all data
m = 2;  % class numbers, class 1: <=50k, class 2: >50k
Y_label_train = Y_total(1:20000,:); % train label
Y_label_test = Y_total(20001:29170,:); % test label

% mu_X_total = mean(X_total,1); %mean of data
% X_total_ = X_total-mu_X_total; %centerize data
A_NUM = 1;
%% Train the algorithm
[n_train,~] = size(X_data_train); %measure the size of train data
[n_test,~] = size(X_data_test); %measure the size of test data

X_ext_train = [X_data_train';ones(1,n_train)]; 
X_ext_test = [X_data_test';ones(1,n_test)]; 

% Prepare parameters
Theta = zeros(d+1,m); %set initial theta
lambda = 0.00001;
convergence = false; 
t = 0; % number of loops
Theta_new = zeros(d+1,m);
Theta_total = zeros(d+1,m,100);
while convergence == false
    A_NUM = A_NUM + 1;
    t = t+1;
%     s_t = 0.01/t;
    s_t = 0.01;
    
%     j = randi(n_train);
%     for k = 1:m
%         p = exp(Theta(:,k)'*X_ext_train(:,j))/(exp(Theta'*X_ext_train(:,j)));
%     end
    
    
    for k = 1:m
        % calaulate df_0
        df_0 = 2*lambda*Theta(:,k);
        H_0 = 2*lambda;
        df_sum = 0;
        H_sum = 0;
        % calaulate sum of df_j
        for j = 1:n_train
            % calaulate p of x_j
            p = exp(Theta(:,k)'*X_ext_train(:,j))/sum(exp(Theta'*X_ext_train(:,j)));    %%%%% sum 怎么只加了一个
            % calaulate df of x_j
            if k == Y_label_train(j)
                df_j = (p-1)*X_ext_train(:,j); % gradient of x_j
            else
                df_j = (p-0)*X_ext_train(:,j); % gradient of x_j
            end

            df_sum = df_sum+df_j; % sum all df_j
            H_sum = H_sum+X_ext_train(:,j)*(p-p^2)*X_ext_train(:,j)'; % sum of all H_j
        end
        df = df_0+df_sum;
        H = H_0+H_sum;

        v = (-1)*(H^(-1)*df);                     %     df-1     H-2

        Theta_new(:,k) = Theta(:,k)+(s_t)*v; %update theta
        
        Theta_total(:,:,t) = Theta_new;
        %compute the CCR 
%         y_hat=zeros(20000,2);
%         yj_hat = zeros(20000,1);
%         for j = 1:20000
%             for l = 1:2
%                y_hat(j,l)=Theta_new(:,l).'*X_ext_train(:,j);
%             end
%             [max_y_hat,index]=max(y_hat(j,:),[],2);
%             yj_hat(j,1)=index;
%         end
%         label_delta = yj_hat- Y_label_train;
%         num=sum(label_delta(:)==0);
%         CCR_train(t, 1) = (1/n_train)*num;
%         
    end
%    Theta = Theta_new;
    if t == 400                          %     
        convergence = true; 
    else
        Theta = Theta_new;
    end
    
%     % whether convergence?
%     delta = sum((Theta_new-Theta).^2,'all');
%     if delta < 1e-8
%         convergence = true; 
%     else
%         Theta = Theta_new;
%     end
end
%%
%CCR_train
y_hat=zeros(20000,2);
yj_hat = zeros(20000,1);
CCR_train = zeros(100,1);
Xaxis = zeros(100,1);
for T = 1:t
    for j = 1:20000
        for l = 1:2
           y_hat(j,l)=Theta_total(:,l,T).'*X_ext_train(:,j);
        end
        [max_y_hat,index]=max(y_hat(j,:),[],2);
        yj_hat(j,1)=index;
    end
    delta_label = yj_hat- Y_label_train;
    num=sum(delta_label(:)==0);
    CCR_train(T,1) = (1/20000)*num;
    Xaxis(T,1) = T;
end
figure(2)
plot(Xaxis,CCR_train);
xlabel('the value of t')
ylabel('the value of CCR ')
title('Newton CCR of the train set');










