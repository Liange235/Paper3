function mu_calc(block)
% This S-function is an auxilarry block for Simulink flowsheets that employ
% the CE-SE solver. It accepts as input a vector containing discrete values
% of a population density function and returns desired moments of the
% crystal size distribution. Optionally, it can also plot the distribution
% function of desired moments.

  setup(block);
  
%endfunction

function setup(block)
  
  %% Register number of input and output ports
  block.NumInputPorts  = 1;
  block.NumOutputPorts = 1;

  %% Setup functional port properties to dynamically
  %% inherited.
  block.SetPreCompInpPortInfoToDynamic;
  block.SetPreCompOutPortInfoToDynamic;
 
  block.InputPort(1).DirectFeedthrough = true;
  
  %% Set block sample time to inherited
  block.SampleTimes = [-1 0];
  
  %% Run accelerator on TLC
  block.SetAccelRunOnTLC(true);
  
  %% Register methods
  block.RegBlockMethod('Outputs',@Output);  
  
  block.NumDialogPrms     = 4;
  block.DialogPrmsTunable = {'Nontunable','Nontunable','Nontunable','Tunable'};
  block.RegBlockMethod('SetOutputPortDimensions', @SetOutPortDims);
  block.RegBlockMethod('SetInputPortDimensions', @SetInpPortDims);
  block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
  block.RegBlockMethod('Start', @Start);
%endfunction

function SetOutPortDims(block, idx, di)
  
 block.OutputPort(idx).Dimensions = di;
 block.InputPort(1).Dimensions    = di;
 
 function SetInpPortDims(block, idx, di)
  
  block.InputPort(idx).Dimensions = di;
  block.OutputPort(1).Dimensions  = length(block.DialogPrm(2).Data);

function DoPostPropSetup(block)
    block.NumDworks = 1;
    block.Dwork(1).Name            = 'param';
    block.Dwork(1).Dimensions      = 4;
    block.Dwork(1).DatatypeID      = 0;      % double
    block.Dwork(1).Complexity      = 'Real'; % real
    block.Dwork(1).UsedAsDiscState = false;
    
function Start(block)
    fname=block.DialogPrm(1).Data;
    init=feval(fname,block,'Init');
    block.Dwork(1).Data=init.param;
    close all
    
function Output(block)

mu_calc=block.DialogPrm(2).Data;
if length(mu_calc>0)
    F=block.InputPort(1).Data;
%   block.OutputPort(1).Data=zeros(length(mu_calc))
    mu_calc=sort(mu_calc);
    
    dL=block.Dwork(1).Data(1);
    
    L_max=dL*length(F);
    
    xx=linspace(0,L_max,L_max/dL);

    cnt=1;
    for i=mu_calc
        FF=F'.*xx.^i;
        block.OutputPort(1).Data(cnt) = trapz(xx,FF);
        cnt=cnt+1;
    end
else
    block.OutputPort(1).Data=0;
end
mu_plot=block.DialogPrm(3).Data;
plot_freq=block.DialogPrm(4).Data;
if length(mu_plot)>0 & rem(block.CurrentTime,plot_freq)==0

    mu_plot=sort(mu_plot);
    F=block.InputPort(1).Data;
    dL=block.Dwork(1).Data(1);
    L_steps=length(F);
    L_max=L_steps*dL;
    xx=linspace(0,L_max,L_steps)';
    cnt=1;
    for i=mu_plot
        FF=F.*xx.^i;
        subplot(length(mu_plot),1,cnt)
        plot(xx,FF)
        a=axis;
        a(3)=0;
        axis(a);
        ylabel (['\mu_' num2str(i) ])
        if cnt ==1
            title('Moment distribution function(s)')
        end
        cnt=cnt+1;
    end
    xlabel('L')

end
%endfunction

