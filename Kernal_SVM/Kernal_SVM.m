data = load('kernel-svm-2rings.mat');
X = data.x;
K = X'*X;

phi = zeros(201,1);

for t =1:t_max
    j = randi(200);
    K(201,201) = 0;
    
end

for i = 1:200
    for j = 1:200
        K(i,j) = X(:,i)'*X(:,j);
    end
end