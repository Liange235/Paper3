function out=pcss_ex_1(block,flag)
persistent p

switch flag
    case 'Init';          p = paramdefine();
                        out = Init(block,p);
    case 'Output';      out = Output(block,p);
    case 'GFdot';       out = GFdot(block,p);
    case 'AuxUpdate';   out = AuxUpdate(block,p);
    otherwise           disp('Error: Unhandled flag');
end

function p = paramdefine()

    dt      = 2; % seconds
    dL      = 10e-6; % microns
    L_max   = 1000e-6; %microns
    a       = 0;
    epsilon = 0.1;
    L_steps = round(L_max/dL);
    xx=linspace(0,L_max,L_max/dL);
    xx3=xx.*xx.*xx;
    p.dL=dL;
    p.dt=dt;
    p.a=a;
    p.epsilon=epsilon;
    p.L_steps=L_steps;
    p.rho_c=2110;
    p.xx=xx;
    p.xx3=xx3;


function out = Init(block,p)

    yy=normpdf(p.xx,200e-6,50e-6)*1e5;
    zz=p.xx3.*yy;
    mu3_0=trapz(p.xx,zz);
    C_0=0.4930;  
    out.IPorts=[1];
    out.OPorts=[p.L_steps 1];  
    out.F0=yy;
    out.FL0=zeros(p.L_steps,1);
    out.G0=zeros(p.L_steps,1);
    out.AuxInit=[C_0 mu3_0];
    out.param=[p.dL p.dt p.a p.epsilon];

function out = Output(block,p)

    out(1).Data = block.Dwork(1).Data;
    out(2).Data = block.Dwork(4).Data(1);

function out = GFdot(block,p)

    T = block.InputPort(1).Data;
    C = block.Dwork(4).Data(1);
    mu3=block.Dwork(4).Data(2);
    Csat= 0.1286 + 0.00588*T + 0.0001721*T*T;
    S=max((C-Csat)/Csat,0);
    G=1.1612e-4 * S^1.32;
    B=4.6401e11*mu3*S^1.78;
    out.G=G*ones(1,p.L_steps);
    out.Fdot=zeros(1,p.L_steps);
    out.GBC=G;
    if G > 0
        out.FBC=B/G;
    else
        out.FBC=0;
    end  

function out = AuxUpdate(block,p)

    F = block.Dwork(1).Data;
    C = block.Dwork(4).Data(1);    
    mu3old=block.Dwork(4).Data(2);
    zz=F.*p.xx3';
    mu3new=trapz(p.xx,zz);
    Cnew=C-(mu3new-mu3old)*p.rho_c;
    out=[Cnew mu3new];