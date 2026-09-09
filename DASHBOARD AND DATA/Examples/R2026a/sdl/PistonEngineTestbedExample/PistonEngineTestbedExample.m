%% Piston Engine Testbed
% 
% This model shows the effect of varying the number of cylinders in a
% piston engine. Four, six, and eight cylinder engines are included with
% firing offsets evenly distributed about their four-stroke cycles. Piston
% pressures are normalized by the number of cylinders to emphasize the
% effect on output vibration.
% 
% 
% 

% Copyright 2008-2025 The MathWorks, Inc.



%% Model

open_system('PistonEngineTestbed')

%% Simulation Results from Simscape Logging
%%
%
% The plots below show engine speeds and cylinder torques for 4, 6, and 8
% cylinder engines.  All engines are acting against the same viscous
% friction load.  The sum of torques produced by all cylinders in each
% engine is also plotted to illustrate the effect of more cylinders on
% engine torque.
%


PistonEngineTestbedPlot1Torque;

%%

