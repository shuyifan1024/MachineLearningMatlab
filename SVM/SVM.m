%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2021
% HW 7
% <shuyi fan><shuyifan@bu.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% import data
ori_data = load('iris.mat');
dataset = [ori_data.X_data_train; ori_data.X_data_test];
dataset = dataset(:,[2 4]);
data_label = [ori_data.Y_label_train; ori_data.Y_label_test];
X_data_test = ori_data.X_data_test;
X_data_train = ori_data.X_data_train;
Y_label_test = ori_data.Y_label_test;
Y_label_train =ori_data.Y_label_train;
X12 = linspace(0,200000,201);
%%
%7.2(a)
data1 = X_data_train((1:35),[2 4]);
data3 = X_data_train((71:105),[2 4]);

label1 = Y_label_train(1:35,1);
label2 = Y_label_train(36:70,1);
label3 = Y_label_train(71:105,1);

data12 = X_data_train((1:70),[2 4]);
data13 = [data1; data3];
data23 = X_data_train((36:105),[2 4]);

label12 = [label1; (label2 - 3)];
label13 = [label1; (label3 - 4)];
label23 = [(label2 - 1); (label3 - 4)];

[W12,g12] = SVM(data12, label12);
[W13,g13] = SVM(data13, label13);
[W23,g23] = SVM(data23, label23);

figure(1)
subplot(3,1,1)
plot(X12,g12);
xlabel('iteration number t')
ylabel('the value of g(theta) ')
title('binary classifier of the dataset fiture 1 & 2 ');
subplot(3,1,2)
plot(X12,g13);
xlabel('iteration number t')
ylabel('the value of g(theta) ')
title('binary classifier of the dataset fiture 1 & 3 ');
subplot(3,1,3)
plot(X12,g23);
xlabel('iteration number t')
ylabel('the value of g(theta) ')
title('binary classifier of the dataset fiture 2 & 3 ');
%%
%7.2(b)
label_b12 = [label1; (label2-3)];
label_b13 = [label1; (label3-4)];
label_b23 = [(label2-1); (label3-4)];
[Yj12,CCR12] = CCR(data12, label_b12);
[Yj13,CCR13] = CCR(data13, label_b13);
[Yj23,CCR23] = CCR(data23, label_b23);

figure(2)
subplot(3,1,1)
plot(X12,CCR12);
xlabel('iteration number t')
ylabel('the value of CCR ')
title('CCR of the train dataset fiture 1 & 2 ');
subplot(3,1,2)
plot(X12,CCR13);
xlabel('iteration number t')
ylabel('the value of CCR ')
title('CCR of the train dataset fiture 1 & 3 ');
subplot(3,1,3)
plot(X12,CCR23);
xlabel('iteration number t')
ylabel('the value of CCR ')
title('CCR of the train dataset fiture 2 & 3 ');
%%
%7.2(c)
data_test1 = X_data_test((1:15),[2 4]);
data_test2 = X_data_test((16:30),[2 4]);
data_test3 = X_data_test((31:45),[2 4]);

label_test1 = Y_label_test((1:15),1);
label_test2 = Y_label_test((16:30),1);
label_test3 = Y_label_test((31:45),1);

label_test12 = [label_test1;(label_test2 -3)];
label_test13 = [label_test1;(label_test3 -4)];
label_test23 = [(label_test2 - 1);(label_test3 -4)];

data_test12 = [data_test1; data_test2];
data_test13 = [data_test1; data_test3];
data_test23 = [data_test2; data_test3];

[Yj_test12,CCR_test12] = CCR(data_test12, label_test12);
[Yj_test13,CCR_test13] = CCR(data_test13, label_test13);
[Yj_test23,CCR_test23] = CCR(data_test23, label_test23);

figure(3)
subplot(3,1,1)
plot(X12,CCR_test12);
xlabel('iteration number t')
ylabel('the value of CCR ')
title('CCR of the test dataset fiture 1 & 2 ');
subplot(3,1,2)
plot(X12,CCR_test13);
xlabel('iteration number t')
ylabel('the value of CCR ')
title('CCR of the test dataset fiture 1 & 3 ');
subplot(3,1,3)
plot(X12,CCR_test23);
xlabel('iteration number t')
ylabel('the value of CCR ')
title('CCR of the test dataset fiture 2 & 3 ');
%%
confusion_train12=confusionmat(Yj12,label12);
confusion_train13=confusionmat(Yj13,label13);
confusion_train23=confusionmat(Yj23,label23);

confusion_test12=confusionmat(Yj_test12,label_test12);
confusion_test13=confusionmat(Yj_test13,label_test13);
confusion_test23=confusionmat(Yj_test23,label_test23);

%%
data_123 = X_data_train(:,[2 4]);
data_test123 = X_data_test(:,[2 4]);
train_ext = [data_123 ones(length(data_123),1)].';
test_ext = [data_test123 ones(length(data_test123),1)].';

Ytrain_hat=zeros(3,105);
Ytest_hat=zeros(3,45);

Ytrain_hat(1,:)= sign(W12(:,200000).'*train_ext);
Ytrain_hat(2,:)= sign(W13(:,200000).'*train_ext);
Ytrain_hat(3,:)= sign(W23(:,200000).'*train_ext);

Ytest_hat(1,:)= sign(W12(:,200000).'*test_ext);
Ytest_hat(2,:)= sign(W13(:,200000).'*test_ext);
Ytest_hat(3,:)= sign(W23(:,200000).'*test_ext);

Ytrain_hat = Ytrain_hat.';
Ytest_hat = Ytest_hat.';
%%


[Yj_train, CCR_train] = all_pairs(Ytrain_hat, Y_label_train);

[Yj_test, CCR_test] = all_pairs(Ytest_hat, Y_label_test);

confusion_train = confusionmat(Yj_train, Y_label_train);
confusion_test  = confusionmat(Yj_test, Y_label_test);
%%
%exam2
data_e = [0 0; 1 1; 1 -1; -1 1; -1 -1];
label_e = [1; -1; -1; -1; -1];
[W_e,g_e] = SVM(data_e, label_e);
[Y_e,CCR_e] = CCR(data_e, label_e);
%%
function [W, g] = SVM(data,label)
    x_ext = [data ones(length(data),1)].';
    t_max = 200000;
    n = length(data);
    theta = [0;0;0];
    W = zeros(3,t_max);
    
    for t = 1:t_max
        st = (1/2)/t;
        C = 1.2;
        j = randi(n);
        
        v = [1 0 0; 0 1 0; 0 0 0]*theta;
        if((label(j,1)*theta.'*x_ext(:,j))<1)
            v = v- n*C*label(j,1)*x_ext(:,j);
        end
        theta = theta - st*v;
        W(:,t) = theta;
    end
    g = zeros(1,201);
    g(1) = C;
   
    for  K = 1:200
        k = K*1000;
        Fj = 0;
        for j = 1:n
            b =(1-label(j,1)*W(:,k).'*x_ext(:,j));
            a=max(0,b);
            fj = C*a;
            Fj = Fj +fj;
        end
        f0 = (1/2)*(W(1,k)^2+W(2,k)^2);
        g(:,K+1) = (1/n)*(f0 +Fj);
       
    end
end
 %%
function [Yj, CCR] = CCR(data, label)
    x_ext = [data ones(length(data),1)].';
    t_max = 200000;
    n = length(data);
    theta = [0;0;0];
    W = zeros(3,t_max);
    CCR = zeros(1,201);
    CCR(1) = 0;
    Yj = zeros(n,1);
    
    for t = 1:t_max
        st = 1/(2*t);
        C = 1.2;
        j = randi(n);
        v = [1 0 0; 0 1 0; 0 0 0]*theta;
        if((label(j,1)*theta.'*x_ext(:,j))<1)
            v = v- n*C*label(j,1)*x_ext(:,j);
        end
        theta = theta - st*v;
        W(:,t) = theta;
    end
    
    for K = 1:200
        total = 0;
        k = 1000*K;
        for j = 1:n
            yj = sign(W(:,k).'*x_ext(:,j));
            if (yj == label(j,1))
                total = total+1;
            end
            Yj(j) = yj;
        end
        CCR(1,K+1) = (1/n)*total;
        
    end    
end
%%
function [yj, CCR] = all_pairs(data, label)
    yj = zeros(length(data),1);
    for i = 1:length(data)
        if data(i,1) ==1 && data(i,2) ==1
            yj(i,1) = 1;
        end
        if data(i,1) == -1 && data(i,3) == 1
            yj(i,1) = 2;
        end
        if data(i,2) == -1 && data(i,3) == -1
            yj(i,1) = 3;
        end
    end
    CCR_total = 0;
    for j = 1:length(data)
        if yj(j,1) == label(j,1)
            CCR_total = CCR_total +1;
        end
    end
    CCR = CCR_total/length(data);
end




















































