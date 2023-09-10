function PCSS(block)

setup(block);
  
%endfunction

function setup(block)

fname=block.DialogPrm(1).Data;
init=feval(fname,block,'Init');
block.NumInputPorts  = length(init.IPorts);
block.NumOutputPorts = length(init.OPorts);

for i=1:length(init.IPorts)
    block.InputPort(i).DatatypeID  = 0;  % double
    block.InputPort(i).Complexity  = 'Real';
    block.InputPort(i).Dimensions    = init.IPorts(i);
    block.InputPort(i).SamplingMode  = 0;
end

for i=1:length(init.OPorts)
    block.OutputPort(i).DatatypeID  = 0; % double
    block.OutputPort(i).Complexity  = 'Real';
    block.OutputPort(i).Dimensions  = init.OPorts(i);
    block.OutputPort(i).SamplingMode  = 0;
end

  block.NumDialogPrms     = 1; % Register parameters
  block.DialogPrmsTunable = {'Nontunable'};
  
  dt=init.param(2);
  block.SampleTimes = [dt 0]; % Register sample times
  block.SetAccelRunOnTLC(false);
  
  block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
  block.RegBlockMethod('Start', @Start);
  block.RegBlockMethod('Outputs', @Outputs);
  block.RegBlockMethod('Update', @Update);
  block.RegBlockMethod('Terminate', @Terminate);

function DoPostPropSetup(block)
block.NumDworks = 5;
init=feval(block.DialogPrm(1).Data,block,'Init');

names=['F  ';'FL ';'G  ';'aux'];
L_steps=length(init.F0);
aux_length=length(init.AuxInit);
lengths=[L_steps L_steps L_steps aux_length];

for i=1:4
  block.Dwork(i).Name            = deblank(names(i,:));
  block.Dwork(i).Dimensions      = lengths(i);
  block.Dwork(i).DatatypeID      = 0;      % double
  block.Dwork(i).Complexity      = 'Real'; % real
  block.Dwork(i).UsedAsDiscState = true;
end
  block.Dwork(5).Name            = 'param';
  block.Dwork(5).Dimensions      = 4;
  block.Dwork(5).DatatypeID      = 0;      % double
  block.Dwork(5).Complexity      = 'Real'; % real
  block.Dwork(5).UsedAsDiscState = false;

%endfunction

function Start(block)
fname=block.DialogPrm(1).Data;
init=feval(fname,block,'Init');

block.Dwork(1).Data = init.F0;
block.Dwork(2).Data = init.FL0;
block.Dwork(3).Data = init.G0;
block.Dwork(4).Data=init.AuxInit;
block.Dwork(5).Data=init.param; % [dL dt a epsilon]

%endfunction

function Outputs(block)
fname=block.DialogPrm(1).Data;
output=feval(fname,block,'Output');

for i=1:length(output)
    block.OutputPort(i).Data=output(i).Data;
end

%endfunction

function Update(block)
fname=block.DialogPrm(1).Data;
%Initialize step variables with previous data
F(1,:)=block.Dwork(1).Data;
FL(1,:)=block.Dwork(2).Data;
G(1,:)=block.Dwork(3).Data;
%Load a few more variables
dL      = block.Dwork(5).Data(1);
dt      = block.Dwork(5).Data(2);
a       = block.Dwork(5).Data(3);
epsilon = block.Dwork(5).Data(4);
L_steps=length(F);
L_max=L_steps*dL;

% get values of G and Fdot
GFdot=feval(fname,block,'GFdot');
Gdat=GFdot.G;
Fdotdat=GFdot.Fdot;
Gshift=([Gdat,0]+[0,Gdat])/2;
G(2,:)=Gshift(2:length(Gshift));
G(3,:)=Gdat;
Fdotshift=([Fdotdat,0]+[0,Fdotdat])/2;
Fdot(1,:)=Fdotshift(2:length(Fdotshift));
Fdot(2,:)=Fdotdat;

% boundary conditions
G(3,1)=GFdot.GBC;
F(3,1)=GFdot.FBC;
FL(3,1)=0;
G(3,L_steps)=0;
F(3,L_steps)=0;
FL(3,L_steps)=0;

% Main loop. Numbers in parenthesis are equation numbers in Chang (1995)
for k=1:2 % loop over t_{n-1/2} and t_{n}
    for j=k:L_steps-1 % loop over every position point except the boundaries      
        % Load some commonly used values into memory
        jp=j+2-k;
        jm=jp-1;
        gp=G(k,jp);
        gm=G(k,jm);
        fp=F(k,jp);
        fm=F(k,jm);
        flp=FL(k,jp);
        flm=FL(k,jm);
        %Calculate F at the current point, following Chang 1995
        dGFdL_p=gp*flp;%(4.10)
        dGFdL_m=gm*flm;%(4.10)         
        dFdt_p=-dGFdL_p;%(4.17)
        dFdt_m=-dGFdL_m;%(4.17)
        dGFdt_p=gp*dFdt_p;%(4.13)
        dGFdt_m=gm*dFdt_m;%(4.13)
        sm=((dL/4)*flm+dt/dL*gm*fm+(dt*dt)/(4*dL)*dGFdt_m);%(4.25)
        sp=((dL/4)*flp+dt/dL*gp*fp+(dt*dt)/(4*dL)*dGFdt_p);%(4.25)
        F(k+1,j)=1/2*(fm+fp+sm-sp)+Fdot(k,j)*dt/2;%(4.24)
        % calculate FL at the current point, following Chang 1995
        f=F(k+1,j);
        Fprime_p=fp+dt/2*dFdt_p;%(4.27)
        Fprime_m=fm+dt/2*dFdt_m;%(4.27)
        FLp=(Fprime_p-f)/(dL/2);%(4.36)
        FLm=-(Fprime_m-f)/(dL/2);%(4.36)
        den=(abs(FLp)^a+abs(FLm)^a);%(4.39)
        if den==0
            W=0;
        else
            W=(abs(FLp)^a*FLm+abs(FLm)^a*FLp)/den;%(4.39)
        end
        dFL=1/2*(flp+flm)-(fp-fm)/dL;%(4.26)
        FL(k+1,j)=(W+(2*epsilon-1)*dFL);%(4.28)
       
    end % End loop over position
end % End loop over time steps
% Update F, FL and G
block.Dwork(1).Data=F(3,:);
block.Dwork(2).Data=FL(3,:);
block.Dwork(3).Data=G(3,:);
%update auxilary states
block.Dwork(4).Data=feval(fname,block,'AuxUpdate');

% end function

function Terminate(block)

%end function
 
