free_data = readmatrix("Dataset2_Fe3pt_zentropy/Fe3Pt_free0_d1_d2.xlsx");
volume_data = free_data(163:end,1);
T_data = free_data(1,2:end)';
free_energy = free_data(163:end,2:end);
T_data = 1+(4/595)*(T_data(1:120)-5);
% T_data = log(1+T_data(1:120));
dx = 6/38;
theta = -0.0408;%-0.0408
kb = 0.1; 
V = []; P = [];T = [];F=[];PP = [];
for n=1:120
x = volume_data(57:134)*cos(atan(theta))+free_energy(57:134,n)*sin(atan(theta));
y = -volume_data(57:134)*sin(atan(theta))+free_energy(57:134,n)*cos(atan(theta));
nx = size(x,1);
yy = y/(kb*T_data(n));
yymin = 0;
V = [V;-3+(6/6)*(x-148)];
T = [T;T_data(n)*ones(nx,1)];
F = [F;y];
P = [P;exp(-y)/sum(exp(-y))];
PP = [PP;exp(-yy+yymin)/sum(dx*exp(-yy+yymin))];

end
VV = reshape(V,78,120);
P1 = reshape(P,78,120);
PP1 = reshape(PP,78,120);
FF = reshape(F,78,120);
dx = 12/78;
Vn = 158.03; % VN normalized volume
%%
Fig2 = figure(2);
clf();
set(gcf,'Position',[207,333,1181,421])
P_CZ = load('CZ_y_pred_Fe3pt_Free_energy.txt');
% P_CZ = load('Fe3pt/CZ_y_pred_Fe3pt_Free_energy.txt');


P_CZ = reshape(P_CZ,78,120);
F_CZ = P_CZ;

e3 = F_CZ-FF;
subplot(2,4,1)
nn=4;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

lg=legend('ZENN','DFT');
lg.Box = 'off';

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,2)
nn=12;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,3)
nn=24;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,4)
nn=32;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,5)
nn=40;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off
subplot(2,4,6)
nn=60;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,7)
nn=90;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,8)
nn=120;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)
xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',14,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off
%%
Fig4 = figure(4);
clf();
set(gcf,'Position',[207,333,1181,421])
P_CZ = load('CZ_y_pred_Fe3pt_Free_energy.txt');

FN = volume_data(57:134)*sin(atan(theta));
P_CZ = reshape(P_CZ,78,120);
F_CZ = P_CZ;

e3 = F_CZ-FF;
subplot(2,4,1)
nn=4;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

lg=legend('Zentropy','DFT');
lg.Box = 'off';

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,2)
nn=12;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,3)
nn=24;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,4)
nn=32;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,5)
nn=40;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off
subplot(2,4,6)
nn=60;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,7)
nn=90;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,4,8)
nn=120;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
xlim([148,160]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')
% title(strcat('T=',num2str(exp(T_data(nn))-1),'K'),'FontSize',12,'FontWeight','bold')
set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
box off


%%
% fig3 =figure(3);
% clf();
gradient = load("CZ_y_pred_Fe3pt_gradient.txt");
TTT = linspace(5,600,596)';
VVV = linspace(148,160,201);
% Vc = [];
% for i=1:196
% y = gradient(:,i);
% zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
% Vc = [Vc VVV(zero_crossings)'];
% end
% plot(TTT(1:55),Vc(1,:)/Vn,'k-','LineWidth',2.0)
% hold on
% plot(TTT(1:55),Vc(2,:)/Vn,'k--','LineWidth',2.0)
% hold on
% plot(TTT(1:55),Vc(3,:)/Vn,'k-','LineWidth',2.0)
% hold on
% % 
% Vc1=[];
% for i=56:164
% y = gradient(:,i);
% zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
% 
% dy = diff(y);  % Compute first derivative
% localmax = find(dy(1:end-1) > 0 & dy(2:end) < 0);  % Find sign changes
% vcvc=[VVV(localmax) VVV(zero_crossings)]';
% Vc1 = [Vc1 vcvc];
% end
% plot(TTT(55:56),[Vc(3,end);Vc1(2,1)]/Vn,'k-','LineWidth',2.0)
% hold on
% plot(TTT(56:164),Vc1(1,:)/Vn,'k--','LineWidth',2.0)
% hold on
% plot(TTT(56:164),Vc1(2,:)/Vn,'k-','LineWidth',2.0)
% hold on
% % % 
% % Vc2 = [];
% % for i=147:150
% % y = gradient(:,i);
% % zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
% % Vc2 = [Vc2 VVV(zero_crossings)'];
% % end
% % plot(TTT(146:147),[Vc1(2,end);Vc2(3,1)],'k-','LineWidth',2.0)
% % hold on
% % plot(TTT(147:150),Vc2(1,:),'k-','LineWidth',2.0)
% % hold on
% % plot(TTT(147:150),Vc2(2,:),'k--','LineWidth',2.0)
% % hold on
% % plot(TTT(147:150),Vc2(3,:),'k-','LineWidth',2.0)
% % hold on
% % plot([TTT(150) TTT(150)],[Vc2(2,end) Vc2(3,end)],'k-','LineWidth',2.0)
% % hold on
% % % % 
% Vc3 = [];
% for i=165:196
% y = gradient(:,i);
% zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
% Vc3 = [Vc3 VVV(zero_crossings)'];
% end
% plot(TTT(164:165),[Vc1(1,end);Vc3(1,1)]/Vn,'k-','LineWidth',2.0)
% hold on
% plot(TTT(165:196),Vc3(1,:)/Vn,'k-','LineWidth',2.0)
% hold on
% % 
% plot(TTT(56),Vc1(1,1)/Vn,'Marker','*','Color','r','MarkerSize',8,'LineWidth',2.0)
% hold on
% plot([TTT(56) TTT(56)],[149 Vc1(1,1)]/Vn,'r-.','LineWidth',2.0)
% hold on
% % plot(TTT(147),Vc2(1,1),'Marker','*','Color','r','MarkerSize',8,'LineWidth',2.0)
% % hold on
% % plot([TTT(147) TTT(147)],[149 Vc2(1,1)],'r-.','LineWidth',2.0)
% % hold on
% plot(TTT(164),Vc1(2,end)/Vn,'Marker','*','Color','r','MarkerSize',8,'LineWidth',2.0)
% hold on
% plot([TTT(164) TTT(164)],[149 Vc1(2,end)]/Vn,'r-.','LineWidth',2.0)
% xlim([5 200])
% ylim([149 154]/Vn)
% xlabel('T (K)')
% ylabel('Volume (V/V_N)')
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% title('P = 6.5 Gpa','FontName','Helvetica','FontSize',12,'FontWeight','bold')
% lgd=legend('stable','unstable');
% lgd.Location = 'best';
% % lgd.Position = [0.148,0.822,0.110,0.102];
% lgd.ItemTokenSize = [30,6];
% lgd.FontWeight = 'bold';
% lgd.Box='off';
% box off
% 
%%
figure(5)
clf();
set(gcf,'Position',[476,446,693,420])
Vn = 158.03;
VVV = VVV/Vn;
Hessian = load("CZ_y_pred_Fe3pt_Hessian.txt");
V1 = [];
df = 0;
for i=1:596
df1 = (df+0.0408)/(df*(-0.0408)+1);
y = gradient(:,i)-df1;
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
V1 = [V1 VVV(zero_crossings)'];
end
plot(V1(1:10:end),TTT(1:10:end),'marker','o','color','k','LineWidth',2.0)
hold on
% 
V4 = [];
df = -0.0187;
for i=1:596
df1 = (df+0.0408)/(df*(-0.0408)+1);
y = gradient(:,i)-df1;
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
if i>155
    VVV1 = min(VVV(zero_crossings));
else
    VVV1 = max(VVV(zero_crossings));
end
V4 = [V4 VVV1];
end
% 
plot(V4(1:20:end)',TTT(1:20:end),'marker','*','color','g','LineWidth',2.0)
hold on
% 
V3 = [];
df = -0.0312;
for i=1:596
df1 = (df+0.0408)/(df*(-0.0408)+1);
y = gradient(:,i)-df1;
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
if i>155
    VVV1 = min(VVV(zero_crossings));
else
    VVV1 = max(VVV(zero_crossings));
end
V3 = [V3 VVV1];
end
% 
plot(V3(1:20:end)',TTT(1:20:end),'marker','square','color','b','LineWidth',2.0)
hold on
% 
V2 = [];
df = -0.0408;%0.0408
for i=1:596
df1 = (df+0.0408)/(df*(-0.0408)+1);
y = gradient(:,i)-df1;
zero_crossings = find(y(1:end-1) .* y(2:end) <= 0);
if i>155
    VVV1 = min(VVV(zero_crossings));
else
    VVV1 = max(VVV(zero_crossings));
end
V2 = [V2 VVV1];
end
% 
plot(V2(1:10:end)',TTT(1:10:end),'marker','diamond','color','r','LineWidth',2.0)
hold on


VH = [];

for i=1:157
y = Hessian(:,i);
zero_crossings = find(y(1:150-1) .* y(2:150) <= 0);
VH = [VH (VVV(zero_crossings(end-1:end))/2+VVV(zero_crossings(end-1:end)+1)/2)'];
end
n = size(VH,2);
plot(VH(1,:),TTT(1:n),'k-','LineWidth',2.0)
hold on
plot([VH(1,end) VH(2,end)],[TTT(n) TTT(n)],'k-','LineWidth',2.0)
hold on
 plot(VH(2,end:-1:1),TTT(n:-1:1),'k-','LineWidth',2.0)

% 1 eV/A^3 = 160.219 GPa

lgd=legend('P = 0.0 GPa','P = 3.0 GPa','P = 5.0 GPa','P = 6.53 GPa');
lgd.Location = 'best';
% lgd.Position = [0.148,0.822,0.110,0.102];
lgd.ItemTokenSize = [10,6];
lgd.FontWeight = 'bold';
lgd.FontSize = 14;
lgd.Box='off';
xlim([148.5/Vn 160/Vn])
ylim([5,600])
ylabel('Temperature (T)')
xlabel('Volume (V/V_N)')
set(gca,'YTick',0:100:600,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off
% % 

%%
fig6=figure(6);
clf();
set(gcf,'Position',[407,94,919,700])
NN =12;
% colors = rand(2*NN,3);
colors = [
    230 159 0;   % Orange
    86 180 233;  % Sky Blue
    0 158 115;   % Bluish Green
    240 228 66;  % Yellow
    0 114 178;   % Blue
    213 94 0;    % Vermillion
    204 121 167; % Reddish Purple
    136 34 85;   % Deep Burgundy
    68 170 153;  % Teal
    153 153 51;  % Olive Green
    17 119 51;   % Dark Green
    136 204 238; % Light Blue
] / 255; % Normalize to range [0,1] for MATLAB colormap

each_con = load('CZ_y_pred_Fe3pt_configuration.txt');
% each_con = load('Fe3pt/CZ_y_pred_Fe3pt_configuration.txt');
config = reshape(each_con,78,120,2*NN);
% subplot(2,2,1)
% for i = 1:NN
% plot((148+(VV(:,nn)+3))/Vn,config(:,1,2*i-1),'color',colors(i,:),'linewidth',2.0);
% hold on
% end
% xlim([148.5/Vn 160/Vn])
% % ylim([5,600])
% ylabel('Each Configuration of Internal Energy (E_i)')
% xlabel('Volume (V/V_N)')
% title ('P = 6.53 Gpa, T = 5 K','FontSize',12,'FontWeight','bold')
% 
% set(gca,'FontName','Helvetica','FontSize',12,'FontWeight','bold','linewidth',1.2)
% box off

subplot(2,2,1)
for i = 1:NN
plot((148+(VV(:,nn)+3))/Vn,config(:,1,2*i).^2,'color',colors(i,:),'linewidth',2.0);
hold on
end
xlim([148.5/Vn 160/Vn])
% ylim([5,600])
ylabel('Each Configuration of Entropy (S_i)')
xlabel('Volume (V/V_N)')
title ('P = 6.53 GPa, T = 5 K','FontSize',14,'FontWeight','bold')
set(gca,'YScale','log','FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,2,2)
for i = 1:NN

plot((148+(VV(:,nn)+3))/Vn,config(:,1,2*i-1)-1*config(:,1,2*i).^2-mean(e3(:,nn))+FN,'color',colors(i,:),'linewidth',2.0);
hold on
end
% plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,1),'k--','linewidth',2.0);
% nn=1;
% plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k--','LineWidth',2.0)


xlim([148.5/Vn 160/Vn])
% ylim([5,600])
ylabel('Each Configuration of Free Energy (F_i)')
xlabel('Volume (V/V_N)')
title ('P = 0 GPa, T = 5 K','FontSize',14,'FontWeight','bold')

set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,2,3)
nn=1;
% plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn)),'k-','LineWidth',2.0)
hold on
% plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn),'r--','LineWidth',2.0)

xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

lg=legend('Zentropy','DFT');
lg.Box = 'off';

title('P = 6.53 GPa, T = 5 K','FontSize',14,'FontWeight','bold')

xlim([148.5/Vn 0.984])
% ylim([5,600])
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(2,2,4)
nn=1;
plot((148+(VV(:,nn)+3))/Vn,F_CZ(:,nn)-mean(e3(:,nn))+FN,'k-','LineWidth',2.0)
hold on
plot((148+(VV(:,nn)+3))/Vn, FF(:,nn)+FN,'r--','LineWidth',2.0)

xlim([148,155]/Vn)
xlabel('Volume (V/V_N)')
ylabel('Free energy (F)')

lg=legend('Zentropy','DFT');
lg.Box = 'off';

title('P = 0 GPa, T = 5 K','FontSize',14,'FontWeight','bold')

xlim([148.5/Vn 160/Vn])
% ylim([5,600])
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

%%
fig7=figure(7);
clf();
set(gcf,'Position',[196,391,1215,304])
subplot(1,3,1)
nn = 12;
free_energy_d2 = free_data_d2(163:end,2:end);
x = volume_data*cos(atan(theta))+free_energy(:,nn)*sin(atan(theta));
ddy = free_energy_d2(:,nn);
dy = free_energy_d1(:,nn);
ddy = ddy.*(1+2*dy*sin(atan(theta))*cos(atan(theta)))./...
    (cos(atan(theta))+dy*sin(atan(theta))).^3;

plot(x/Vn,ddy,'k-','LineWidth',2.0)
hold on
plot((148+(linspace(-3,9,201)+3))/Vn,Hessian(:,56),'r--','LineWidth',2.0)
hold on
plot([0.8 1],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',1.2)

xlim([148,154]/Vn)
xlabel('Volume (V/V_N)')
ylabel('\partial^2 F/\partial V^2')
lg=legend('Zentropy','DFT');
lg.Location = 'best';
lg.Box = 'off';

title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')

% ylim([5,600])
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(1,3,2)
nn = 24;
free_energy_d2 = free_data_d2(163:end,2:end);
x = volume_data*cos(atan(theta))+free_energy(:,nn)*sin(atan(theta));
ddy = free_energy_d2(:,nn);
dy = free_energy_d1(:,nn);
ddy = ddy.*(1+2*dy*sin(atan(theta))*cos(atan(theta)))./...
    (cos(atan(theta))+dy*sin(atan(theta))).^3;

plot(x/Vn,ddy,'k-','LineWidth',2.0)
hold on
plot((148+(linspace(-3,9,201)+3))/Vn,Hessian(:,116),'r--','LineWidth',2.0)
hold on
plot([0.8 1],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',1.2)

xlim([148,154]/Vn)
xlabel('Volume (V/V_N)')
ylabel('\partial^2 F/\partial V^2')


title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')

% ylim([5,600])
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off

subplot(1,3,3)
nn = 33;
free_energy_d2 = free_data_d2(163:end,2:end);
x = volume_data*cos(atan(theta))+free_energy(:,nn)*sin(atan(theta));
ddy = free_energy_d2(:,nn);
dy = free_energy_d1(:,nn);
ddy = ddy.*(1+2*dy*sin(atan(theta))*cos(atan(theta)))./...
    (cos(atan(theta))+dy*sin(atan(theta))).^3;

plot(x/Vn,ddy,'k-','LineWidth',2.0)
hold on
plot((148+(linspace(-3,9,201)+3))/Vn,Hessian(:,161),'r--','LineWidth',2.0)
hold on
plot([0.8 1],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',1.2)

xlim([148,154]/Vn)
ylim([-0.001,0.01])
xlabel('Volume (V/V_N)')
ylabel('\partial^2 F/\partial V^2')


title(strcat('T=',num2str(5+(595/4)*(T_data(nn)-1)),'K'),'FontSize',12,'FontWeight','bold')

% ylim([5,600])
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off
%%
fig8=figure(8);
clf();
set(gcf,'Position',[493,632,309,234])

nn = 33;
free_energy_d2 = free_data_d2(163:end,2:end);
x = volume_data*cos(atan(theta))+free_energy(:,nn)*sin(atan(theta));
ddy = free_energy_d2(:,nn);
dy = free_energy_d1(:,nn);
ddy = ddy.*(1+2*dy*sin(atan(theta))*cos(atan(theta)))./...
    (cos(atan(theta))+dy*sin(atan(theta))).^3;

plot(x/Vn,ddy,'k-','LineWidth',2.0)
hold on
nn = 32;
free_energy_d2 = free_data_d2(163:end,2:end);
x = volume_data*cos(atan(theta))+free_energy(:,nn)*sin(atan(theta));
ddy = free_energy_d2(:,nn);
dy = free_energy_d1(:,nn);
ddy = ddy.*(1+2*dy*sin(atan(theta))*cos(atan(theta)))./...
    (cos(atan(theta))+dy*sin(atan(theta))).^3;

plot(x/Vn,ddy,'r--','LineWidth',2.0)
hold on
plot([0.8 1],[0 0],'color',[0.8,0.8,0.8],'LineStyle','--','LineWidth',1.2)

xlim([0.955,0.961])
ylim([-0.4,1]*10^-3)
xlabel('Volume (V/V_N)')
ylabel('\partial^2 F/\partial V^2')
lg=legend('T = 165K','T = 160K');
lg.Location = 'best';
lg.Box = 'off';


% ylim([5,600])
set(gca,'FontName','Helvetica','FontSize',14,'FontWeight','bold','linewidth',1.2)
box off