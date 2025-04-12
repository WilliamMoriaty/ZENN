% fig1=figure(1);
% clf();
% set(gcf,"Position",[285,243,711,349])
% F_CZ = load('CZ_y_pred_triple_T7.txt');
% F_CE = load('CE_y_pred_triple_T7.txt');
% kb = 1;
% TT=linspace(0.5,6,23);
% x_vals = linspace(-6,6,300);
% V = @(x,T) (1/6*0.06)*T*(x.^6)-(4.0*0.06*T)*x.^4+(24/5.5*(T-0.5)+12)*T*0.06*(x.^2);
% PP = []
% for i=1:23
% P = exp(-V(x_vals,TT(i))/(kb*TT(i)));
% P = P./trapz(x_vals,P,2);
% PP = [PP P'];
% end
% 
% subplot(2,3,1)
% plot(x_vals,-log(PP(:,7)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,7)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,7)),'b--','LineWidth',1.2)
% 
% title(strcat('T=',num2str(TT(7))),'FontSize',12,'FontWeight','bold')
% ylim([0,8])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% lgd = legend("Benchmark","Zentropy","Entropy");
% %lgd.Location = 'best';
% lgd.Position = [0.186,0.616,0.094,0.130];
% lgd.ItemTokenSize = [10,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
% 
% subplot(2,3,2)
% plot(x_vals,-log(PP(:,11)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,11)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,11)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(11))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,3)
% plot(x_vals,-log(PP(:,15)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,15)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,15)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(15))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% 
% subplot(2,3,4)
% plot(x_vals,-log(PP(:,9)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,9)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,9)),'b--','LineWidth',1.2)
% 
% ylim([0,6])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(9))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,5)
% plot(x_vals,-log(PP(:,10)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,10)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,10)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(10))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,6)
% plot(x_vals,-log(PP(:,12)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,12)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,12)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(12))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% 
% %%
% 
% fig1=figure(1);
% clf();
% set(gcf,"Position",[285,243,711,349])
% F_CZ = load('CZ_y_pred_triple_T7.txt');
% F_CE = load('CE_y_pred_triple_T7.txt');
% 
% TT=linspace(0.5,6,23);
% x_vals = linspace(-6,6,300);
% V = @(x,T) (1/6*0.06)*T*(x.^6)-(4.0*0.06*T)*x.^4+(24/5.5*(T-0.5)+12)*T*0.06*(x.^2);
% PP = []
% for i=1:23
% P = exp(-V(x_vals,TT(i))/(kb*TT(i)));
% P = P./trapz(x_vals,P,2);
% PP = [PP P'];
% end
% 
% subplot(2,3,1)
% plot(x_vals,-log(PP(:,7)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,7)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,7)),'b--','LineWidth',1.2)
% 
% title(strcat('T=',num2str(TT(7))),'FontSize',12,'FontWeight','bold')
% ylim([0,8])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% lgd = legend("Benchmark","Zentropy","Entropy");
% %lgd.Location = 'best';
% lgd.Position = [0.186,0.616,0.094,0.130];
% lgd.ItemTokenSize = [10,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
% 
% subplot(2,3,2)
% plot(x_vals,-log(PP(:,11)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,11)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,11)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(11))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,3)
% plot(x_vals,-log(PP(:,15)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,15)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,15)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(15))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% 
% subplot(2,3,4)
% plot(x_vals,-log(PP(:,9)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,9)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,9)),'b--','LineWidth',1.2)
% 
% ylim([0,6])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(9))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,5)
% plot(x_vals,-log(PP(:,10)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,10)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,10)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(10))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,6)
% plot(x_vals,-log(PP(:,12)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,12)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,12)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(12))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% %%
% fig1=figure(1);
% clf();
% set(gcf,"Position",[285,243,711,349])
% F_CZ = load('CZ_y_pred_triple_T7.txt');
% F_CE = load('CE_y_pred_triple_T7.txt');
% 
% TT=linspace(0.5,6,23);
% x_vals = linspace(-6,6,300);
% V = @(x,T) (1/6*0.06)*T*(x.^6)-(4.0*0.06*T)*x.^4+(24/5.5*(T-0.5)+12)*T*0.06*(x.^2);
% PP = []
% for i=1:23
% P = exp(-V(x_vals,TT(i))/(kb*TT(i)));
% P = P./trapz(x_vals,P,2);
% PP = [PP P'];
% end
% 
% subplot(2,3,1)
% plot(x_vals,-log(PP(:,7)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,7)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,7)),'b--','LineWidth',1.2)
% 
% title(strcat('T=',num2str(TT(7))),'FontSize',12,'FontWeight','bold')
% ylim([0,8])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% lgd = legend("Benchmark","Zentropy","Entropy");
% %lgd.Location = 'best';
% lgd.Position = [0.186,0.616,0.094,0.130];
% lgd.ItemTokenSize = [10,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
% 
% subplot(2,3,2)
% plot(x_vals,-log(PP(:,11)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,11)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,11)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(11))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,3)
% plot(x_vals,-log(PP(:,15)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,15)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,15)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(15))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% 
% subplot(2,3,4)
% plot(x_vals,-log(PP(:,9)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,9)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,9)),'b--','LineWidth',1.2)
% 
% ylim([0,6])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(9))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,5)
% plot(x_vals,-log(PP(:,10)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,10)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,10)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(10))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,6)
% plot(x_vals,-log(PP(:,12)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,12)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,12)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(12))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% 
% %%
% 
% fig2=figure(2);
% clf();
% set(gcf,"Position",[285,243,711,349])
% F_CZ = load('CZ_y_pred_T7.txt');
% F_CE = load('CE_y_pred_T7.txt');
% 
% TT=linspace(0.5,6,23);
% x_vals = linspace(-4,4,200);
% V = @(x,T) (1/8*T)*(x.^4-8*x.^2)+(-0.4*T*(T-1)+0.2*T)*x;
% PP = []
% for i=1:23
% P = exp(-V(x_vals,TT(i))/(kb*TT(i)));
% P = P./trapz(x_vals,P,2);
% PP = [PP P'];
% end
% 
% subplot(2,3,1)
% plot(x_vals,-log(PP(:,5)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,5)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,5)),'b--','LineWidth',1.2)
% 
% title(strcat('T=',num2str(TT(5))),'FontSize',12,'FontWeight','bold')
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% lgd = legend("Benchmark","Zentropy","Entropy");
% %lgd.Location = 'best';
% lgd.Position = [0.143,0.790,0.094,0.130];
% lgd.ItemTokenSize = [10,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
% 
% subplot(2,3,2)
% plot(x_vals,-log(PP(:,11)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,11)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,11)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(11))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,3)
% plot(x_vals,-log(PP(:,15)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,15)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,15)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(15))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% 
% subplot(2,3,4)
% plot(x_vals,-log(PP(:,4)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,4)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,4)),'b--','LineWidth',1.2)
% 
% ylim([0,6])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(4))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,5)
% plot(x_vals,-log(PP(:,9)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,9)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,9)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(9))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% 
% subplot(2,3,6)
% plot(x_vals,-log(PP(:,12)),'k-','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CZ(:,12)),'r-.','LineWidth',1.2)
% hold on
% plot(x_vals,-log(F_CE(:,12)),'b--','LineWidth',1.2)
% ylim([0,5])
% xlabel('x')
% ylabel('Free energy (F/(kb\timesT))')
% title(strcat('T=',num2str(TT(12))),'FontSize',12,'FontWeight','bold')
% 
% box off
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)

%%
fig3=figure(3);
clf();
set(gcf,"Position",[459,196,615,446])
F_CZ = load('y_pred_bifurcation.txt');
F_CE = load('CE_y_pred_bifurcation.txt');
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
subplot(2,2,1)
plot(x_vals,-log(P(:,11)),'k-','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CZ(:,11)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CE(:,11)),'b--','LineWidth',2.0)

title(strcat('T =',num2str(T_vals(11),1)),'FontSize',14,'FontWeight','bold')
ylim([2.5,3.4])
xlabel('x')
ylabel('Free Energy ( F / k_BT )')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

lgd = legend("Benchmark","ZENN","DNN");
lgd.Location = 'best';
% lgd.Position = [0.147715447154472,0.824242152466368,0.299186991869919,0.102];
lgd.ItemTokenSize = [25,6];
lgd.FontWeight = 'bold';
lgd.Box='off';

subplot(2,2,2)
plot(x_vals,-log(P(:,25)),'k-','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CZ(:,25)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CE(:,25)),'b--','LineWidth',2.0)
xlim([-1.5,1.5])
ylim([1.7,2.1])
xlabel('x')
ylabel('Free Energy ( F / k_BT )')
title(strcat('T =',num2str(T_vals(25),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

subplot(2,2,3)
plot(x_vals,-log(P(:,37)),'k-','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CZ(:,37)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CE(:,37)),'b--','LineWidth',2.0)
ylim([2.05,2.15])
xlabel('x')
ylabel('Free Energy ( F / k_BT )')
title(strcat('T =',num2str(T_vals(37),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

% 
subplot(2,2,4)
plot(x_vals,-log(P(:,50)),'k-','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CZ(:,50)),'r-.','LineWidth',2.0)
hold on
plot(x_vals,-log(F_CE(:,50)),'b--','LineWidth',2.0)
xlim([-0.6,0.6])
ylim([2.25,2.32])
xlabel('x')
ylabel('Free Energy ( F / k_BT )')
title(strcat('T =',num2str(T_vals(50),2)),'FontSize',14,'FontWeight','bold')

box off
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)

