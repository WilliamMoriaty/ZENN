% 初始化参数
clc
clear
num_elements = 3;  % 9 维概率向量
T_values = linspace(1, 8, 100);  % T 从 1 变化到 3
p_matrix = zeros(num_elements, length(T_values)); % 存储不同 T 下的 p
T = T_values;
% % 系数调节以满足 P1(1) > 0.6 且 P1 > P2 > P3
% a1 = 2.0;
% a2 = 1.0;
% a3 = 0.8;
% 
% f1 = a1 * exp(-T);               % 指数下降
% f2 = a2 * exp(-(T - 2).^2);      % 高斯
% f3 = a3 * exp(T - 3);            % 指数上升
% 
% Z = f1 + f2 + f3;
% 
% P1 = f1 ./ Z;
% P2 = f2 ./ Z;
% P3 = f3 ./ Z;
f1 = 2.0 * exp(-(T - 2).^2) + 1.5 * exp(-(T - 6).^2);
f2 = 2.5 * exp(-(T - 3).^2) + 1.2 * exp(-(T - 5.5).^2);
f3 = 2.2 * exp(-(T - 4).^2) + 1.4 * exp(-(T - 6.5).^2);

% ---- 归一化为概率 ----
Z = f1 + f2 + f3;
P1 = f1 ./ Z;
P2 = f2 ./ Z;
P3 = f3 ./ Z;
p_matrix = [P1;P2;P3];

% 绘制随 T 变化的概率分布
figure;
plot(T_values, p_matrix', 'LineWidth', 2);
xlabel('T');
ylabel('p');
title('Probability Vector p Changing with T');
grid on;
legend(arrayfun(@(x) sprintf('p_{%d}', x), 1:num_elements, 'UniformOutput', false));

% TT= [T_values(1) T_values(25) T_values(50) T_values(75) T_values(100)];
% t = [1 25 50 75 100];
t = 1:4:100;
TT = T_values(t);
data = [];
for j=1:size(t,2)

n = 1e4; % Number of numbers to generate
random_numbers = generate_numbers_by_probability(p_matrix(:,t(j)), n);

%  one-hot vector
true_labels_onehot = zeros(n, num_elements);
for i = 1:n
    true_labels_onehot(i, random_numbers(i)) = 1;
end
yy1 = true_labels_onehot;
yy1 = [yy1 ones(n,1)*TT(j)];
data = [data;yy1];
end

writematrix(data,'Multiclass_data.csv')


