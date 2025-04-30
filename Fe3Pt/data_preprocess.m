free_data = readmatrix("Dataset2_Fe3pt_zentropy/Fe3Pt_free0_d1_d2.xlsx");
volume_data = free_data(163:end,1);
T_data1 = free_data(1,2:end)';
free_energy = free_data(163:end,2:end);
T_data = 1+(4/595)*(T_data1(1:120)-5);
% T_data = log(1+T_data1(1:120));
dx = 6/38;
theta = -0.0408;% P=6.53 GPa
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
V_T = [V T];
nn = 1;
subplot(1,3,1)
plot(VV(:,nn), FF(:,nn),'k-');
subplot(1,3,2)
plot(VV(:,nn),kb*T_data(nn)*(-log(PP1(:,nn))),'r-')
subplot(1,3,3)
plot(VV(:,nn),((PP1(:,nn))),'r-')

writematrix(V_T,'V_T.csv')
writematrix(PP,'PP.csv')
