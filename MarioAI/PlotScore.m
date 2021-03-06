%% PlotScore.m
clear all
close all
clc


x = 0:50:50*39;
load ../train_easy_eval_easy_seed_default/eval.txt
eval_easy = eval(1:40,:);
load ../train_medium_eval_medium_seed7801/eval.txt
eval_medium = eval;


% Plot total score
figure(1)
grid on
hold on
plot(x,eval_easy(:,1),'b-o')
plot(x,eval_medium(:,1),'r-o')
p = polyfit(x, eval_easy(:,1)', 1);
plot(x,x*p(1)+p(2), 'm--');
p = polyfit(x, eval_medium(:,1)', 1);
plot(x,x*p(1)+p(2), 'm--');
hold off

legend('Level 0 easy', ...
       'Level 0 hard', 'Location','Best')

xlabel('Training iterations');
ylabel('Average Evaluation Scores');
print -depsc2 score_figure
print -djpeg100 score_figure
%%
figure(2)
plot(x,eval_easy(:,2),'b-o');
hold on
plot(x,eval_medium(:,2),'r-o');



p = polyfit(x, eval_easy(:,2)', 1);
plot(x,x*p(1)+p(2), 'm--');
p = polyfit(x, eval_medium(:,2)', 1);
plot(x,x*p(1)+p(2), 'm--');



grid on
legend('Successful Rate: Level 0 easy', ...
       'Successful Rate: Level 0 hard', ...
       'Location','Best')
xlabel('Training iterations');
ylabel('Average Rate');
print -depsc2 success_figure
print -djpeg100 success_figure
%%
figure(3)
plot(x,eval_easy(:,3),'b-s');
hold on
plot(x,eval_medium(:,3),'r-s');

grid on
legend('Killing %: Level 0 easy', ...
       'Killing %: Level 0 hard', ...  
     'Location','Best')
xlabel('Training iterations');
ylabel('Average Rate');
print -depsc2 killing_figure
print -djpeg100 killing_figure
%%
figure(4)
plot(x,eval_easy(:,5),'b-x');
hold on
plot(x,eval_medium(:,5),'r-x');

grid on
legend('Time spent: Level 0 easy', ...
       'Time spent: Level 0 hard', ...  
     'Location','Best')
xlabel('Training iterations');
ylabel('Time ticks');
print -depsc2 time_figure
print -djpeg100 time_figure

