
fig1=figure(1);
clf();
F = load('y_pred_dw_10.txt');
x = linspace(-4,4,100);
T = linspace(0.1,6,100);

colormap('jet')
surf(T, x, -log(F), 'EdgeColor', 'none','FaceColor','interp');  % Surface plot
hold on;
zlim([0,8])
clim([0,8])
contour(T, x, -log(F), 10);  % Contour lines
view(-65,50)
xlabel('T','FontSize',12,'FontWeight','bold','Position',[2.95,-5.31,0.21])
ylabel('x','FontSize',12,'FontWeight','bold','Position',[-0.85,0.27,-0.88])
set(gca,'FontWeight','bold','FontSize',12,'LineWidth',1.2)

ch = colorbar;
ch.Label.String='Free Energy (F/(k_B\timesT))';
ch.Label.FontSize=14;
ch.Label.FontWeight='bold';
ch.Location ='eastoutside';

% fig2=figure(2);
% clf();
% F = load('y_pred_tw.txt');
% x = linspace(-6,6,100);
% T = linspace(0.5,6,100);
% 
% 
% colormap('jet')
% surf(T, x, -log(F), 'EdgeColor', 'none','FaceColor','interp');  % Surface plot
% hold on;
% zlim([-1,10])
% clim([-1,10])
% contour(T, x, -log(F), 10);  % Contour lines
% view(-54,60)
% xlabel('T','FontSize',12,'FontWeight','bold','Position',[3.08,-7.45,-0.27])
% ylabel('x','FontSize',12,'FontWeight','bold','Position',[-0.19,0.08,-0.28])
% set(gca,'FontWeight','bold','FontSize',12,'LineWidth',1.2)
% 
% ch = colorbar;
% ch.Label.String='Free Energy (F/(kb\timesT))';
% ch.Label.FontSize=14;
% ch.Label.FontWeight='bold';
% ch.Location ='eastoutside';
% 

fig3=figure(3);
clf();
F = load('y_pred_bifurcation.txt');
x_min = -2;
x_max = 2;
T_min = 0.1;
T_max = 4;
N = 100;  % Number of points

% Generate grid for x and temperature
x_vals = linspace(x_min, x_max, N);
T_vals = linspace(T_min, T_max, N);
dx = x_vals(2)-x_vals(1);
dT = T_vals(2)-T_vals(1);

% F = F*(dx*dT*sum(sum(F)));
F = -log(F);
colormap('jet')
surf(T_vals, x_vals, F, 'EdgeColor', 'none','FaceColor','interp');  % Surface plot
hold on;
zlim([1.5,2.5])
clim([1.5,2.5])
contour(T_vals, x_vals, F, 200,'ZLocation','zmin','LineWidth',1.2);  % Contour lines
view(-42,45)
xlabel('T','FontSize',15,'FontWeight','bold','Position',[2.09,-2.56,1.40])
ylabel('x','FontSize',15,'FontWeight','bold','Position',[-0.438,0.068,1.42])
set(gca,'FontWeight','bold','FontSize',15,'LineWidth',1.2)

ch = colorbar;
ch.Label.String='Free Energy ( F / k_BT )';
ch.Label.FontSize=15;
ch.Label.FontWeight='bold';
ch.Location ='eastoutside';