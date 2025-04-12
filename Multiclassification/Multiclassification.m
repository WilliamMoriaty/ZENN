fig1=figure(1);
clf();
set(gcf,'Position',[284,280,751,464])

load('p_matrix.mat')
CZ = load('CZ_multiclass_prob.txt');
CZ = CZ';

CE = load('CE_multiclass_prob.txt');
CE = CE';

T_values = linspace(1, 8, 100);  % T 从 0.1 变化到 7，避免数值问题
TT1= T_values(1:4:73);
TT2= T_values(76:3:100);

t1 = 1:4:73;
t2 = 76:3:100;
subplot(2,2,1)

plot(T_values, p_matrix', 'LineWidth', 2);
hold on
plot([6.1,6.1],[0,1],'Color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)
xlim([1,7.5])
xlabel('Temperature (T)')
ylabel('Probability (p)')
text(3.0,0.95,'Class 1','FontSize',12,'FontWeight','bold','Color',[0.00,0.45,0.74])
text(3.0,0.85,'Class 2','FontSize',12,'FontWeight','bold','Color',[0.85,0.33,0.10])
text(3.0,0.75,'Class 3','FontSize',12,'FontWeight','bold','Color',[0.93,0.69,0.13])

set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

for j=1:3
subplot(2,2,j+1)
plot(TT1,p_matrix(j,t1),'Marker','o','MarkerSize',8,'Color',[17 119 51]/255,'LineStyle','none','LineWidth',2.0)
hold on
plot(TT2,p_matrix(j,t2),'Marker','o','MarkerSize',8,'Color','k','LineStyle','none','LineWidth',2.0)
hold on
plot(T_values,CZ(j,:),'Color','r','LineStyle','-','LineWidth',2.0)
hold on
plot(T_values,CE(j,:),'Color','b','LineStyle','-','LineWidth',2.0)
hold on
plot([6.1,6.1],[0,1],'Color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',2.0)
if j==1
lgd=legend('Training data','Predictive data','ZENN','DNN');
% lgd.Location = 'best';
lgd.Position = [0.673,0.780,0.131,0.128];
lgd.ItemTokenSize = [10,6];
lgd.FontWeight = 'bold';
lgd.Box='off';
end
xlabel('Temperature (T)')
ylabel('Probability (p)')
xlim([1,7.5])
title(strjoin({append('Class', blanks(1)), num2str(j)}, ''),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off
end
