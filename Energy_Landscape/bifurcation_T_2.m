fig3=figure(3);
clf();
set(gcf,"Position",[459,196,615,446])
F_CZ = load('y_pred_bifurcation_10.txt');
F_CE = load('CE_y_pred_bifurcation_10.txt');
% Define x and temperature range
x_min = -2;
x_max = 2;
T_min = 0.1;
T_max = 4;
N = 100;  % Number of points
T_vals = linspace(T_min, T_max, N);
x_vals = linspace(x_min,x_max,N);
[X, T] = meshgrid(x_vals, T_vals);
F = (X.^2 / 2 + (T - 2) / 2).^2 + ((T - 2).^2 - 1).^2 / 2;
V = F .* T;  % Scaling with temperature

kb = 1;  % Boltzmann constant
P = exp(-V ./ (kb * T));  % Unnormalized probability
dx = (x_max-x_min)/(N-1);
dT = (T_max-T_min)/(N-1);
P=P'/(dx*dT*sum(sum(P)));
% F_CZ = F_CZ*(dx*dT);
% F_CZ = -log(F_CZ);
CZ = [];CE = [];
E_CZ = abs(F_CZ-P);
CZ=[max(max(E_CZ)) CZ];
E_CE = abs(F_CE-P);
CE=[max(max(E_CE)) CE];

F_CZ = load('y_pred_bifurcation.txt');
F_CE = load('CE_y_pred_bifurcation.txt');
E_CZ = abs(F_CZ-P);
CZ=[max(max(E_CZ)) CZ];
E_CE = abs(F_CE-P);
CE=[max(max(E_CE)) CE];

F_CZ = load('y_pred_bifurcation_80.txt');
F_CE = load('CE_y_pred_bifurcation_80.txt');
E_CZ = abs(F_CZ-P);
CZ=[max(max(E_CZ)) CZ];
E_CE = abs(F_CE-P);
CE=[max(max(E_CE)) CE];
bar([CE;CZ]')

xlabel('Sample Number')
ylabel('max |p_{error}|')
% box off
set(gca,'XTickLabel',[6400,1600,100],...
    'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
box off
lgd = legend("DNN (4 layer with 48 neurons)","ZENN (K=6,2 layer with 8 neurons (E_k,S_k))");
lgd.Location = 'west';
% lgd.Position = [0.148,0.822,0.110,0.102];
lgd.ItemTokenSize = [10,6];
lgd.FontWeight = 'bold';
lgd.Box='off';

