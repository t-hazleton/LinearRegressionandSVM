%read in delivery data
DT = readtable('DeliveryTimes.xls');


%%
%assign first column to y and last 2 colums to x
x = DT(1:25,2:3);
y = DT(1:25,1);

%convert them to matrix of doubles
x = table2array(x);
y = table2array(y);
figure(1);
subplot(3,3,1)
text(0.35,0.5,'Time','fontsize',14);
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

subplot(3,3,2)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
hold on;
plot(x(:,1),y(:,1),'k.');
hold off;

subplot(3,3,3);
hold on;
plot(x(:,2),y(:,1),'k.');
hold off;
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

subplot(3,3,4);
hold on;
plot(y(:,1),x(:,1),'k.');
hold off;
set(gca,'XTickLabel',[]);

subplot(3,3,5)
text(0.3,0.5,'Cases','fontsize',14);
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

subplot(3,3,6);
hold on;
plot(x(:,1),x(:,2),'k.');
hold off;
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

subplot(3,3,7);
hold on;
plot(x(:,2),y(:,1),'k.');
hold off;

subplot(3,3,8);
hold on;
plot(x(:,2),x(:,1),'k.');
hold off;
set(gca,'YTickLabel',[]);

subplot(3,3,9)
text(0.25,0.5,'Distance','fontsize',14);
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

x1 = ones(25,1);
x = horzcat(x1,x);
%%
%equation to find beta
Beta = inv(x'*x)*(x'*y);
%%
%Mean Squared Error MSE = sqrt[1/N*summation((y_pre-y_act)^2)]
%mdl = fitlm(x,y);
%ysim = random(mdl);
ysim = x*Beta;
r = 0;
N = length(y);
%this is the summation of y predicted and y actual
for i = 1:25
    r =  r+((y(i,1) - ysim(i,1))^2);
end
%r = r*r;
MSE =sqrt((1/N)*r);











