% Code to plot simulation results from PistonEngineTestbed
%% Plot Description:
%
% The plots below show engine speeds and cylinder torques for 4, 6, and 8
% cylinder engines.  All engines are acting against the same viscous
% friction load.  The sum of torques produced by all cylinders in each
% engine is also plotted to illustrate the effect of more cylinders on
% engine torque.

% Copyright 2016 The MathWorks, Inc.

% Generate simulation results if they don't exist
if ~exist('simlog_PistonEngineTestbed', 'var')
    sim('PistonEngineTestbed')
end

% Reuse figure if it exists, else create new figure
if ~exist('h1_PistonEngineTestbed', 'var') || ...
        ~isgraphics(h1_PistonEngineTestbed, 'figure')
    h1_PistonEngineTestbed = figure('Name', 'PistonEngineTestbed');
end
figure(h1_PistonEngineTestbed)
clf(h1_PistonEngineTestbed)

% Get simulation results: Engine Speeds
simlog_t = simlog_PistonEngineTestbed.Inertia_4_Cylinder.w.series.time;
simlog_wEngine4Cyl = simlog_PistonEngineTestbed.Inertia_4_Cylinder.w.series.values('rpm');
simlog_wEngine6Cyl = simlog_PistonEngineTestbed.Inertia_6_Cylinder.w.series.values('rpm');
simlog_wEngine8Cyl = simlog_PistonEngineTestbed.Inertia_8_Cylinder.w.series.values('rpm');

% Get simulation results: Torque from individual pistons
simlog_num_cylinders = length(simlog_PistonEngineTestbed.Engine_4_Cylinder.piston);
simlog_4CylTrq = zeros(length(simlog_t),simlog_num_cylinders);
for i=1:simlog_num_cylinders
    simlog_4CylTrq = simlog_PistonEngineTestbed.Engine_4_Cylinder.piston(i).t.series.values('N*m');
end

simlog_num_cylinders = length(simlog_PistonEngineTestbed.Engine_6_Cylinder.piston);
simlog_6CylTrq = zeros(length(simlog_t),simlog_num_cylinders);
for i=1:simlog_num_cylinders
    simlog_6CylTrq = simlog_PistonEngineTestbed.Engine_6_Cylinder.piston(i).t.series.values('N*m');
end

simlog_num_cylinders = length(simlog_PistonEngineTestbed.Engine_8_Cylinder.piston);
simlog_8CylTrq = zeros(length(simlog_t),simlog_num_cylinders);
for i=1:simlog_num_cylinders
    simlog_8CylTrq = simlog_PistonEngineTestbed.Engine_8_Cylinder.piston(i).t.series.values('N*m');
end

% Plot results
simlog_handles(1) = subplot(2, 2, 1);
plot(simlog_t, sum(simlog_wEngine4Cyl,2), 'LineWidth', 1)
hold on
plot(simlog_t, sum(simlog_wEngine6Cyl,2), 'LineWidth', 1)
plot(simlog_t, sum(simlog_wEngine8Cyl,2), 'LineWidth', 1)
hold off
grid on
title('Engine Speed')
ylabel('Speed (RPM)')
legend({'4 Cyl','6 Cyl','8 Cyl'},'Location','Best');

simlog_handles(2) = subplot(2, 2, 2);
plot(simlog_t, sum(simlog_4CylTrq,2), 'LineWidth', 3)
hold on
plot(simlog_t, simlog_4CylTrq, 'LineWidth', 1)
hold off
grid on
title('Torques from 4 Cylinders')
ylabel('Torque (N*m)')
temp_ylim(:,1) = get(gca,'YLim');
legend('Sum','Location','Best');

simlog_handles(3) = subplot(2, 2, 3);
plot(simlog_t, sum(simlog_6CylTrq,2), 'LineWidth', 3)
hold on
plot(simlog_t, simlog_6CylTrq, 'LineWidth', 1)
hold off
grid on
title('Torques from 6 Cylinders')
ylabel('Torque (N*m)')
temp_ylim(:,2) = get(gca,'YLim');
xlabel('Time (s)')
legend('Sum','Location','Best');

simlog_handles(4) = subplot(2, 2, 4);
plot(simlog_t, sum(simlog_8CylTrq,2), 'LineWidth', 3)
hold on
plot(simlog_t, simlog_8CylTrq, 'LineWidth', 1)
hold off
grid on
title('Torques from 8 Cylinders')
ylabel('Torque (N*m)')
temp_ylim(:,3) = get(gca,'YLim');
xlabel('Time (s)')
legend('Sum','Location','Best');

% Match Y limits, symmetrically about 0
temp_maxYlim = max(max(temp_ylim));
set(simlog_handles(2:4),'YLim',[-1 1]*temp_maxYlim);

linkaxes(simlog_handles, 'x')

% Remove temporary variables
clear simlog_t simlog_handles

clear simlog_wEngine4Cyl simlog_wEngine6Cyl simlog_wEngine8Cyl 
clear simlog_4CylTrq simlog_6CylTrq simlog_8CylTrq
clear temp_maxYlim temp_pistoni temp_ylim simlog_num_cylinders
