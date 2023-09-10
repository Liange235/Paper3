function out=pcss_ex_2(block,flag)
persistent param

switch flag
    case 'Init';        param = paramdefine(); 
                        out = Init(block,param);
    case 'Output';      out = Output(block,param);
    case 'GFdot';       out = GFdot(block,param);
    case 'AuxUpdate';   out = AuxUpdate(block,param);
    otherwise           disp('Error: Unhandled flag');
end

function param = paramdefine()
    dt      = 1; %min
    dL      = 0.01; %mm
    L_max   = 4; %mm
    a       = 0;
    epsilon = 0.05;
    L_f     = 0.3; % mm
       
    L_steps=floor(L_max/dL);
    L_max=dL*L_steps;
    xx=linspace(0,L_max,L_steps);
    yy=zeros(1,L_steps);
    yy(1)=1;
    xx3=xx.*xx.*xx;
    s=round(L_f/dL);
    H_f=[zeros(1,s) ones(1,L_steps-s)];
    
    param(1).name='dL';       param(1).Data=dL;       param(1).units='mm';
    param(2).name='dt';       param(2).Data=dt;       param(2).units='min';
    param(3).name='a';        param(3).Data=a;        param(3).units='none';
    param(4).name='epsilon';  param(4).Data=epsilon;  param(4).units='none';
    param(5).name='L steps';  param(5).Data=L_steps;  param(5).units='none';
    param(6).name='Rho_c';    param(6).Data=1989;     param(6).units='g/l';
    param(7).name='B data';   param(7).Data=yy;       param(7).units='none';
    param(8).name='x_data';   param(8).Data=xx;       param(8).units='mm';
    param(9).name='x_data^3'; param(9).Data=xx3;      param(9).units='mm^3';
    param(10).name='k_v';     param(10).Data=0.11120; param(10).units='none';
    param(11).name='L_max';   param(11).Data=L_max;   param(11).units='mm';
    param(12).name='V';       param(12).Data=10.5;    param(12).units='l';
    param(13).name='Csat';    param(13).Data=4.038;   param(13).units='mol/l';
    param(14).name='Hf_data'; param(14).Data=H_f;     param(14).units='none';
    

function [out param] = Init(block,param)
    
    dL      = param(1).Data;
    L_steps = param(5).Data;
    xx      = param(8).Data;
    L_max   = param(11).Data;
    
    load PCSS_ex_2_data F FL G c

    out.IPorts=[1 1 1];
    out.OPorts=[L_steps 1];
    out.F0=F;
    out.FL0=FL;
    out.G0=G;    
    
    out.AuxInit=[c 0 0];
    out.param=[param(1).Data param(2).Data param(3).Data param(4).Data];

function out = Output(block,param)
    F       = block.Dwork(1).Data;
    c       = block.Dwork(4).Data(1);

    out(1).Data=F;
    out(2).Data=c;

function out = GFdot(block,param)

    c_in    = block.InputPort(1).Data;
    q       = block.InputPort(2).Data; 
    R       = block.InputPort(3).Data;
    F       = block.Dwork(1).Data;
    FL      = block.Dwork(2).Data;
    c       = block.Dwork(4).Data(1);
    dL      = param(1).Data;
    L_steps = param(5).Data;
    rho_c   = param(6).Data; 
    B_data  = param(7).Data;
    xx      = param(8).Data; 
    xx3     = param(9).Data; 
    k_v     = param(10).Data;
    L_max   = param(11).Data; 
    V       = param(12).Data; 
    Csat    = param(13).Data; 
    H_f     = param(14).Data;
    k_g     = 3.0513e-2;
    g       = 1;
    k_b     = 8.357e9;
    b       = 4;
    
    h_f=R*(1-H_f);
    tau=V/q;
    S=max(c-Csat,0);
    G=k_g * S^g;
    B=k_b*S^b*1e-6; %convert from L to mm^3
    Fdot=-F'./tau.*(h_f+1);
   
    out.G=G* ones(1,L_steps);
    out.Fdot=Fdot;
    out.GBC=G;
    if S > 0;
        out.FBC=B/G;
    else
        out.FBC = 0;
    end
    xx2=xx.*xx;
    mu2=trapz(xx2.*F')*dL;
    mu3=trapz(xx,xx3.*F');
    dedt=-3*k_v*k_g * S^g*mu2+q/V*k_v*trapz(xx,xx3.*F'.*(h_f+1));
    block.Dwork(4).Data(2)=dedt;
    e=1-k_v*mu3;
    block.Dwork(4).Data(3)=e;

function out =AuxUpdate(block,param)

    c_in    = block.InputPort(1).Data;
    q       = block.InputPort(2).Data;
    R       = block.InputPort(3).Data;
    F       = block.Dwork(1).Data;
    c       = block.Dwork(4).Data(1);
    dedt    = block.Dwork(4).Data(2);
    e       = block.Dwork(4).Data(3);    
    dL      = param(1).Data;
    dt      = param(2).Data;
    L_steps = param(5).Data;
    rho_c   = param(6).Data;
    xx      = param(8).Data;
    xx3     = param(9).Data;
    L_max   = param(11).Data;
    V       = param(12).Data;
    H_f     = param(14).Data;

    M_A=74.551;
    h_f=R*(1-H_f);
    T1=q*(rho_c-M_A*c)/V;
    T2=(rho_c-M_A*c)/e*dedt;
    T3=q*M_A*c_in/(V*e);
    T4=-q*rho_c/(V*e);
    dcdt=1/M_A*(T1+T2+T3+T4);
    Cnew=c+dcdt*dt;
    out=[Cnew dedt e];