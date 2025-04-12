
gradient = load("CZ_y_pred_bifurcation_gradient.txt");
Hessian = load("CZ_y_pred_bifurcation_Hessian.txt");
gradient_CE = load("CE_y_pred_bifurcation_gradient.txt");
Hessian_CE = load("CE_y_pred_bifurcation_Hessian.txt");

TTT = linspace(0.1,4,100)';
VVV = linspace(-2,2,100);
fig3 =figure(3);
clf();
set(gcf,'Position',[454,462,389,212])

%% Benchmark
plot([0,0],[0,2],'b--','LineWidth',2.0)
hold on
plot([0,0],[2,4],'b-','LineWidth',2.0)
hold on
plot(-sqrt(2-linspace(0.1,2,50)),linspace(0.1,2,50),'b-','LineWidth',2.0)
hold on
plot(sqrt(2-linspace(0.1,2,50)),linspace(0.1,2,50),'b-','LineWidth',2.0)
hold on
% % 
%% zentropy
Vc = [];
for i=1:49
y = gradient(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc = [Vc (VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end

%%
Vc1 = [];
for i=50:50
y = gradient(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc1 = [Vc1 ones(3,1)*(VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end
% 

plot(Vc(1,:),TTT(1:49),'k-','LineWidth',2.0)
hold on
plot(Vc(2,:),TTT(1:49),'k--','LineWidth',2.0)
hold on
%%
% % plot(Vc1(1,1),TTT(50),'Marker','*','Color','r','MarkerSize',8,'LineWidth',2.0,'LineStyle','none')
% % hold on
% %%
plot(Vc(3,:),TTT(1:49),'k-','LineWidth',2.0)
hold on
% % 
% % % 
plot([Vc(1,end) Vc1(1,1)],TTT(49:50),'k-','LineWidth',2.0)
hold on
plot([Vc(2,end) Vc1(2,1)],TTT(49:50),'k--','LineWidth',2.0)
hold on
plot([Vc(3,end) Vc1(3,1)],TTT(49:50),'k-','LineWidth',2.0)
hold on
% % 
% %   
Vc3 = [];
for i=51:100
y = gradient(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc3 = [Vc3 (VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end
plot([Vc1(3,end);Vc3(1,1)],TTT(50:51),'b-','LineWidth',2.0)
hold on
plot(Vc3(1,:),TTT(51:100),'k-','LineWidth',2.0)
hold on
% % % % 
% % % plot(Vc1(1,1),TTT(50),'Marker','*','Color','r','MarkerSize',8,'LineWidth',2.0)
% % % hold on
% % 
% % % plot([-1.5 Vc1(1,1)],[TTT(50) TTT(50)],'r-.','LineWidth',2.0)
% % % hold on
% % 
%% entropy
Vc_CE = [];
for i=1:43
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE = [Vc_CE (VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end
 
Vc_CE1 = [];
for i=44:47
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE1 = [Vc_CE1 (VVV(zero_crossings(1:2:5))'+VVV(zero_crossings(1:2:5)+1)')/2];
end

Vc_CE = [Vc_CE Vc_CE1];

Vc_CE1 = [];
for i=48:50
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE1 = [Vc_CE1 (VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end
Vc_CE = [Vc_CE Vc_CE1];

Vc_CE1 = [];
for i=51:51
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE1 = [Vc_CE1 ones(3,1).*(VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end
% 
plot(Vc_CE(1,:),TTT(1:50),'LineStyle','-','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
plot(Vc_CE(2,:),TTT(1:50),'LineStyle','--','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
% % %%
% % % plot(Vc_CE1(3,1),TTT(49),'Marker','*','Color','r','MarkerSize',8,'LineWidth',2.0,'LineStyle','none')
% % % hold on
plot(Vc_CE(3,:),TTT(1:50),'LineStyle','-','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
plot([Vc_CE(1,end) Vc_CE1(1,1)],TTT(50:51),'LineStyle','-','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
plot([Vc_CE(2,end) Vc_CE1(2,1)],TTT(50:51),'LineStyle','--','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
plot([Vc_CE(3,end) Vc_CE1(3,1)],TTT(50:51),'LineStyle','-','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
% % plot([Vc_CE1(1,end) Vc_CE1(3,1)],[TTT(49),TTT(49)],'LineStyle','--','LineWidth',2.0,'Color',[230 159 0]/255)
% % 
Vc_CE3 = [];
for i=52:54
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE3 = [Vc_CE3 (VVV(zero_crossings)'+VVV(zero_crossings+1)')/2];
end

Vc_CE4 = [];
for i=55:58
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE4 = [Vc_CE4 (VVV(zero_crossings(2))'+VVV(zero_crossings(2)+1)')/2];
end

Vc_CE5 = [];
for i=59:100
y = gradient_CE(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
Vc_CE5 = [Vc_CE5 (VVV(zero_crossings(1))'+VVV(zero_crossings(1)+1)')/2];
end

Vc_CE5 =[Vc_CE3 Vc_CE4 Vc_CE5];
plot(Vc_CE5(1,:),TTT(52:100),'LineStyle','-','LineWidth',2.0,'Color',[230 159 0]/255)
hold on
% % 
% % 
% % 
text(0.8,2.2,'Benchmark','FontSize',14,'FontWeight','bold','Color','b')
text(0.8,2.0,'ZENN','FontSize',14,'FontWeight','bold','Color','k')
text(0.8,1.8,'DNN','FontSize',14,'FontWeight','bold','Color',[230 159 0]/255)

ylim([0.5 2.5])
xlim([-1.5 1.5])
ylabel('T')
xlabel('x ')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.5)
% title('P = 6.5 Gpa','FontName','Helvetica','FontSize',12,'FontWeight','bold')
% lgd=legend('stable','unstable','critical point');
% lgd.Location = 'best';
% % lgd.Position = [0.148,0.822,0.110,0.102];
% lgd.ItemTokenSize = [30,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
box off
% % 

%%
fig4 =figure(4);
clf();
set(gcf,'Position',[454,462,389,212])

VH = [];

for i=1:48
y = Hessian(:,i);
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
if y(1)>0
VH = [VH (VVV(zero_crossings(1:2))/2+VVV(zero_crossings(1:2)+1)/2)'];
else
VH = [VH (VVV(zero_crossings(2:3))/2+VVV(zero_crossings(2:3)+1)/2)'];
end
end

VH = [VH [VVV(50) VVV(50)]'];
n = size(VH,2);

%%
VH_CE = [];

for i=1:50
y = Hessian_CE(:,i);
yy = gradient_CE(:,i);
zero_gradient = find(yy(1:end-1) .* yy(2:end) <= 0);
ymin = min(zero_gradient);
ymax = max(zero_gradient);

zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
aa = find(zero_crossings<ymax & zero_crossings>ymin);


VH_CE = [VH_CE (VVV([zero_crossings(aa(1)) zero_crossings(aa(end))])/2+VVV([zero_crossings(aa(1)) zero_crossings(aa(end))]+1)/2)'];

end


VH_CE = [VH_CE [VVV(50) VVV(50)]'];
n2 = size(VH_CE,2);
%% real
xx = linspace(-sqrt(2/3),sqrt(2/3),51);
T_real = 2-3*xx.^2;
plot(xx,T_real,'LineStyle','--','Color','b','LineWidth',2.0)
hold on
%%
% plot(VVV(49),TTT(51),'Marker','o','markersize',8,'LineStyle','none','Color','r','LineWidth',1.5)

%% zentropy
plot(VH(1,:),TTT(1:n),'k-','LineWidth',2.0)
hold on
plot([VH(1,end) VH(2,end)],[TTT(n) TTT(n)],'k-','LineWidth',2.0)
hold on
plot(VH(2,end:-1:1),TTT(n:-1:1),'k-','LineWidth',2.0)
hold on
% plot([-1.5,VVV(49)],[TTT(51) TTT(51)],'LineStyle','--','Color','r','LineWidth',1.5)
% hold on
% plot([VVV(49),VVV(49)],[0 TTT(51)],'LineStyle','--','Color','r','LineWidth',1.5)
% hold on
%% entropy
plot(VH_CE(1,:),TTT(1:n2),'LineStyle','-','LineWidth',2.0,'color',[230 159 0]/255)
hold on
plot([VH_CE(1,end) VH_CE(2,end)],[TTT(n2) TTT(n2)],'LineStyle','-','LineWidth',2.0,'color',[230 159 0]/255)
hold on
plot(VH_CE(2,end:-1:1),TTT(n2:-1:1),'LineStyle','-','LineWidth',2.0,'color',[230 159 0]/255)
hold on

text(-0.5,2.8,'Benchmark','FontSize',14,'FontWeight','bold','Color','b')
text(-0.5,2.5,'ZENN','FontSize',14,'FontWeight','bold','Color','k')
text(-0.5,2.2,'DNN','FontSize',14,'FontWeight','bold','Color',[230 159 0]/255)

% lgd=legend('Benchmark','critical point','zentropy');
% lgd.Location = 'best';
% % lgd.Position = [0.148,0.822,0.110,0.102];
% lgd.ItemTokenSize = [30,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
% box off

%
xlim([-1 1])
ylim([0.5,3])
ylabel('T')
xlabel('x')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.5)
box off
%%
fig5=figure(5);
clf();
set(gcf,"Position",[459,196,615,446])

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
% P = F';
P = exp(-V ./ (kb * T));  % Unnormalized probability
dx = (x_max-x_min)/(N-1);
dT = (T_max-T_min)/(N-1);
P=P'/(dx*dT*sum(sum(P)));
% F_CZ = F_CZ*(dx*dT);
% F_CZ = -log(F_CZ);
FFxx = @(x,T)3*x.^2+T-2;
subplot(2,2,1)
plot(x_vals,FFxx(x_vals,T_vals(11)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(Hessian(:,11)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(Hessian_CE(:,11)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

title(strcat('T =',num2str(T_vals(11),1)),'FontSize',14,'FontWeight','bold')
xlim([-1,1])
ylim([-2,2])
xlabel('x')
ylabel('\partial^2 F/\partial x^2')
box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

lgd = legend("Benchmark","ZENN","DNN");
lgd.Location = 'best';
% lgd.Position = [0.147715447154472,0.824242152466368,0.299186991869919,0.102];
lgd.ItemTokenSize = [30,6];
lgd.FontWeight = 'bold';
lgd.Box='off';

subplot(2,2,2)
plot(x_vals,FFxx(x_vals,T_vals(25)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(Hessian(:,25)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(Hessian_CE(:,25)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

xlim([-1,1])
ylim([-2,2])
xlabel('x')
ylabel('\partial^2 F/\partial x^2')
title(strcat('T =',num2str(T_vals(25),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

subplot(2,2,3)
plot(x_vals,FFxx(x_vals,T_vals(37)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(Hessian(:,37)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(Hessian_CE(:,37)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

xlim([-1,1])
ylim([-1,2])
xlabel('x')
ylabel('\partial^2 F/\partial x^2')
title(strcat('T =',num2str(T_vals(37),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

% 
subplot(2,2,4)
plot(x_vals,FFxx(x_vals,T_vals(50)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(Hessian(:,50)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(Hessian_CE(:,50)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

xlim([-1,1])
ylim([-1,2])
xlabel('x')
ylabel('\partial^2 F/\partial x^2')
title(strcat('T =',num2str(T_vals(50),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

%%
fig5=figure(6);
clf();
set(gcf,"Position",[459,196,615,446])

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
% P = F';
P = exp(-V ./ (kb * T));  % Unnormalized probability
dx = (x_max-x_min)/(N-1);
dT = (T_max-T_min)/(N-1);
P=P'/(dx*dT*sum(sum(P)));
% F_CZ = F_CZ*(dx*dT);
% F_CZ = -log(F_CZ);
Fx = @(x,T)2*x.*(x.^2/2+(T-2)/2);
subplot(2,2,1)
plot(x_vals,Fx(x_vals,T_vals(11)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(gradient(:,11)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(gradient_CE(:,11)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

title(strcat('T =',num2str(T_vals(11),1)),'FontSize',14,'FontWeight','bold')
xlim([-1.5,1.5])
% ylim([-2,2])
xlabel('x')
ylabel('\partial F/\partial x')
box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

lgd = legend("Benchmark","ZENN","DNN");
lgd.Location = 'best';
% lgd.Position = [0.147715447154472,0.824242152466368,0.299186991869919,0.102];
lgd.ItemTokenSize = [30,6];
lgd.FontWeight = 'bold';
lgd.Box='off';

subplot(2,2,2)
plot(x_vals,Fx(x_vals,T_vals(25)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(gradient(:,25)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(gradient_CE(:,25)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

xlim([-1.2,1.2])
% ylim([-2,2])
xlabel('x')
ylabel('\partial F/\partial x')
title(strcat('T =',num2str(T_vals(25),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

subplot(2,2,3)
plot(x_vals,Fx(x_vals,T_vals(37)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(gradient(:,37)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(gradient_CE(:,37)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

xlim([-1,1])
% ylim([-1,2])
xlabel('x')
ylabel('\partial F/\partial x')
title(strcat('T =',num2str(T_vals(37),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

% 
subplot(2,2,4)
plot(x_vals,Fx(x_vals,T_vals(50)),'k-','LineWidth',2.0)
hold on
plot(x_vals,(gradient(:,50)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,(gradient_CE(:,50)),'b--','LineWidth',2.0)
hold on
plot([-2 2],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)

xlim([-1,1])
% ylim([1,4])
xlabel('x')
ylabel('\partial F/\partial x')
title(strcat('T =',num2str(T_vals(50),2)),'FontSize',14,'FontWeight','bold')

set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off