function csd = run_sim(mv)
% start_time = '4';
% normal_stop_time = '11';
    filename = 'pcss_example_2';
	L_max = 4;
    L_steps = 400;
    p = mv;
    in = Simulink.SimulationInput(filename);
	in = in.setModelParameter('AbsTol', '1e-5', ...
                          'StopTime', '5000');
	nn = strcat('[',num2str(p(1)),',', num2str(p(2)),',', num2str(p(3)),']');
    in = in.setBlockParameter('pcss_example_2/[c_in   q   R]   ss','Value', nn);
	simOut = sim(in);
	F = simOut.yout.signals(4).values(end,:)';
    xx=linspace(0,L_max,L_steps)';
    mu3_L = F.*xx.^3;
    csd = mu3_L;
    disp('Simulation completed.');
