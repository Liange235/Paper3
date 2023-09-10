clear
clc
filename = 'pcss_example_2';
load("PCSS_ex_2_data.mat");
% start_time = '4';
% normal_stop_time = '11';
L_max = 4;
L_steps = 400;
bd_l = [4.1, 0.05, 1.0];
bd_u = [5.0, 0.09, 3.0];
n = 3;
p = lhsdesign(n, 3);
p = bd_l+repmat(bd_u-bd_l, n, 1).*p;
input = zeros(n ,3);
output = zeros(n ,400);
in = Simulink.SimulationInput(filename);
in = in.setModelParameter('AbsTol', '1e-5', ...
                          'StopTime', '5000');
for i =1:n
    nn = strcat('[',num2str(p(i,1)),',', num2str(p(i,2)),',', num2str(p(i,3)),']');
% simOut = sim('filename','ReturnWorkspaceOutputs','on');
% load_system(filename)
% cs = getActiveConfigSet(filename);
% set_param(filename, ...
%             'AbsTol','1e-5',...
%              'StopTime', '3000', ... 
%              'SaveTime','on','TimeSaveName','tout', ...
%              'SaveState','on','StateSaveName','xoutNew',...
%              'SaveOutput','on','OutputSaveName','youtNew',...
%              'SignalLogging','on','SignalLoggingName','logsout');
    in = in.setBlockParameter('pcss_example_2/[c_in   q   R]   ss','Value', nn);
    simOut = sim(in);
    F = simOut.yout.signals(4).values(end,:)';
    xx=linspace(0,L_max,L_steps)';
    mu3_L = F.*xx.^3;
    input(i,:) = p(i,:);
    output(i,:) = mu3_L;
    disp(strcat('Current iteration is : ',num2str(i)));
end