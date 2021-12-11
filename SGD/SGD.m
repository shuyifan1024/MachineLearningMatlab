data = load('iris.mat');
dataset = [data.X_data_train; data.X_data_test];
data_label = [data.Y_label_train; data.Y_label_test];
X_data_test = data.X_data_test;
X_data_train = data.X_data_train;
Y_label_test = data.Y_label_test;
Y_label_train =data.Y_label_train;
%%
figure(1)
histogram(data_label);
xlabel('class label')
ylabel('the number of every class ')
title('the scatter of every class ');
%%
means = mean(dataset);
delta = dataset - means;
coef = zeros(4,4);
for i = 1:4
    for j = 1:4
        if i<=j
            cov = delta(:,i)' * delta(:,j);
            var = prod(((sum(delta(:,i).^2,1))*(sum(delta(:,j).^2,1))).^(1/2));
            coef(i,j) = cov / var;
        else
            coef(i,j) = nan;
        end
    end  
end

%%
figure(2)
subplot(3,2,1)
scatter (dataset(:,1), dataset(:,2));
xlabel('the value of fiture 1')
ylabel('the value of fiture 2 ')
title('2D scatter plots of the dataset fiture 1 & 2 ');
subplot(3,2,2)
scatter (dataset(:,1), dataset(:,3));
xlabel('the value of fiture 1')
ylabel('the value of fiture 3 ')
title('2D scatter plots of the dataset fiture 1 & 3 ');
subplot(3,2,3)
scatter (dataset(:,1), dataset(:,4));
xlabel('the value of fiture 1')
ylabel('the value of fiture 4')
title('2D scatter plots of the dataset fiture 1 & 4 ');
subplot(3,2,4)
scatter (dataset(:,2), dataset(:,3));
xlabel('the value of fiture 2')
ylabel('the value of fiture 3 ')
title('2D scatter plots of the dataset fiture 2 & 3 ');
subplot(3,2,5)
scatter (dataset(:,2), dataset(:,4));
xlabel('the value of fiture 2')
ylabel('the value of fiture 4 ')
title('2D scatter plots of the dataset fiture 2 & 4 ');
subplot(3,2,6)
scatter (dataset(:,3), dataset(:,4));
xlabel('the value of fiture 3')
ylabel('the value of fiture 4 ')
title('2D scatter plots of the dataset fiture 3 & 4 ');

%%
La = 0.1;    % Lambda
n = length(X_data_train);
one = ones(105,1);
x_ext = [X_data_train one].';
H = zeros(5, 3);
P = zeros(3,1);
V = zeros(5,3);
H_T = zeros(5,3,300); % H total with 300 data
g = zeros(300,1);
for t = 1:6000
    j = randi(n);
    yj = Y_label_train(j,1);
    % P theta
    % Vk
    for k = 1:3
        P(k,1)=exp(H(:,k).'*x_ext(:,j))/(exp(H(:,1).'*x_ext(:,j))+exp(H(:,2).'*x_ext(:,j))+exp(H(:,3).'*x_ext(:,j)));
        if(k==yj)
            V(:,k) = 2*La*H(:,k) + n*(P(k,1)-1)*x_ext(:,j);
        else
            V(:,k) = 2*La*H(:,k) + n*P(k,1)*x_ext(:,j);
        end
    end
    % Theta k
    for k =1:3
        st = 0.01/t;
        H(:,k) = H(:,k) - st*V(:,k);
    end
    % H Theta
    if (mod(t,20)==0)
        a = t/20;
        H_T(:,:,a) = H;
    end
end
Xaxis = zeros(300,1);
for J = 1:300
    
    Xaxis(J,1)=J*20;
    Fj = 0;
    f0 = La * (H_T(:,1,J).'* H_T(:,1,J)+H_T(:,2,J).'* H_T(:,2,J)+H_T(:,3,J).'* H_T(:,3,J));
    
    for c = 1:105
        fj = log(exp(H_T(:,1,J).'*x_ext(:,c))+exp(H_T(:,2,J).'*x_ext(:,c))+exp(H_T(:,3,J).'*x_ext(:,c)));
        for L =1:3
            if (L == Y_label_train(c))
                fj = fj - H_T(:,L,J).'*x_ext(:,c);
            end
        end
        Fj = Fj + fj;
    end
    g(J,1) = (1/n)*(f0 +Fj);
    
end
figure(3)
plot (Xaxis,g);
xlabel('iteration number t')
ylabel('the value of 1/n * gΘ ')
title('L2-regularized logistic loss');
%%%%%%%%%%%%%%     randi

%%
% CCR train
y_hat=zeros(105,3);
yj_hat = zeros(105,1);
CCR_train = zeros(300,1);
for t = 1:300
    for j = 1:105
        for l = 1:3
           y_hat(j,l)=H_T(:,l,t).'*x_ext(:,j);
        end
        [max_y_hat,index]=max(y_hat(j,:),[],2);
        yj_hat(j,1)=index;
    end
    delta = yj_hat- Y_label_train;
    num=sum(delta(:)==0);
    CCR_train(t,1) = (1/n)*num;
end
figure(4)
plot(Xaxis,CCR_train);
xlabel('the value of t')
ylabel('the value of CCR ')
title('CCR of the train set');

%%
% CCR test
y_test_hat=zeros(45,3);
yj_test_hat = zeros(45,1);
one = ones(45,1);
x_test_ext = [X_data_test one].';
CCR_test = zeros(300,1);
for t = 1:300
    for j = 1:45
        for l = 1:3
           y_test_hat(j,l)=H_T(:,l,t).'*x_test_ext(:,j);
        end
        [max_y_hat,index]=max(y_test_hat(j,:),[],2);
        yj_test_hat(j,1)=index;
    end
    delta = yj_test_hat- Y_label_test;
    num=sum(delta(:)==0);
    CCR_test(t,1) = (1/45)*num;
end
figure(5)
plot(Xaxis,CCR_test);
xlabel('the value of t')
ylabel('the value of CCR ')
title('CCR of the test set');

%%
% logloss
logloss = zeros(300,1);
for t = 1:300
    P_total = 0;
    for j = 1:45
        k = Y_label_test(j);
        P_log=log(exp(H_T(:,k,t).'*x_test_ext(:,j))/(exp(H_T(:,1,t).'*x_test_ext(:,j))+exp(H_T(:,2,t).'*x_test_ext(:,j))+exp(H_T(:,3,t).'*x_test_ext(:,j))));
        P_total = P_total +P_log;
    end
    logloss(t,1)= (-1/45)*P_total; 
end
figure(6)
plot(Xaxis,logloss);
xlabel('the value of t')
ylabel('the value of logloss ')
title('log-loss of the test set');
%%
% 6.3 f
% confusion matrix
confusion_train=confusionmat(yj_hat,Y_label_train);
confusion_test=confusionmat(yj_test_hat,Y_label_test);

%%
% 6.3 g
sub = 1;
figure(7)
for i = 1:4
    for j =1:4
        if i<j
            data = x_ext;
            [Xgrid, Ygrid] = meshgrid([(min(data(i,:))):0.1:(max(data(i,:)))],[(min(data(j,:))):0.1:(max(data(j,:)))]);
            new_data = zeros(length(Xgrid(:)), 5);
            new_data(:, i) = Xgrid(:);
            new_data(:, j) = Ygrid(:);
            new_data(:, 5) = 1;
            new_data = new_data.';
            yj_a = zeros(length(Xgrid(:)),1);
            y_p = zeros(length(Xgrid(:)),1);
            for c = 1:length(Xgrid(:))
                
                for l = 1:3
                    y_p(c,l)=H_T(:,l,300).'*new_data(:,c);
                end
                [max_y_hat,index]=max(y_p(c,:),[],2);
                yj_a(c,1)=index;
            end
            subplot(3,2,sub)
            gscatter( Xgrid(:), Ygrid(:),yj_a);
            xlabel(['the feature of ',num2str(i)])
            ylabel(['the feature of ',num2str(j)])
            title(['2-dimensional subsets of feature',num2str(i),' & feature', num2str(j)]);
            sub = sub+1;
        end
    end
end
%%



























