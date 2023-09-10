clear
clc
close all
% load('CSD_Y.mat');
% load('CSD_X.mat');
output = load("MWD_Y.txt");
input = load("MWD_X.txt");
% output = output(1:2000,:);
% input = input(1:2000,:);
chain = load("chain.txt");
n = length(output);
dis = zeros(n, 1);
for i = 1:1:n
    a = output(i, :);
    b = zeros(n, 1);
    for j = 1:1:n
           b(j) = norm(a- output(j,:));
    end
    dis(i) = max(b);
end
[val, ind] = max(dis);
target_x = input(ind, :);
target = output(ind, :);
dis_ind = zeros(n, 1);
for k = 1:1:n
    dis_ind(k) = norm(target- output(k, :));
end
[val_st, ind_st] = max(dis_ind);
start_x = input(ind_st, :);
start = output(ind_st, :);
figure(1)
plot(chain', start);
hold on 
plot(chain', target);
legend('s', 't')
% path = 'C:\Users\t1714\Desktop\Academic\Coding_Files\Experiment Design\MWD_single_condition\Result\Continuous crystalization process';
path = 'C:\Users\t1714\Desktop\Academic\Coding_Files\Experiment Design\MWD_single_condition\Result\Ethylene homo-polymerization';
% dir = strcat(path, '\csd_ini30.txt');
% dir_new = strcat(path, '\csd_ini32.txt');
dir = strcat(path, '\mwd_ini30.txt');
dir_new = strcat(path, '\mwd_ini32.txt');
data = load(dir);
% data_new = vertcat(data, start, target);
data_new = vertcat(data, target, start);
fid = fopen(dir_new,'wt');
for i = 1:32
    fprintf(fid, '%.19e\t', data_new(i, :));
    fprintf(fid, '\n');
end
dirx = strcat(path, '\mv_ini30.txt');
dirx_new = strcat(path, '\mv_ini32.txt');
datax = load(dirx);
% datax_new = vertcat(datax, start_x, target_x);
datax_new = vertcat(datax, target_x, start_x);
fid = fopen(dirx_new,'wt');
for i = 1:32
    fprintf(fid, '%.19e\t', datax_new(i, :));
    fprintf(fid, '\n');
end
