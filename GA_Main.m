%% GA_Main
clr;
%% Data preprocessing
global X DFunc
% input data to the observations of the n-by-p data matrix X
rng default; % For reproducibility
X = [randn(100,3)*0.75+ones(100,3);
    randn(100,3)*0.5-ones(100,3);
    randn(100,3).^2];

% raw data visualization
figure;
plot3(X(:,1),X(:,2),X(:,3),'.');
title 'Randomly Generated Data';
%% GA parameters
% Fitness function
FitnessFunction = @(x)ObjectFnc(x);
% Number of Decision Variable
nvars = 3; %1: k number of class, 2: Distance Metric, 3: Number of times to repeat clustering using new initial cluster centroid positions
DFunc = {'sqeuclidean', 'cityblock', 'cosine', 'correlation'};
% Solution boundary
lb = [2 1 5];
ub = [5.99 4.99 20.99];
%% User defines GA options
options = optimoptions(@gamultiobj,'PlotFcn',{@gaplotpareto,@gaplotscorediversity},...
'CreationFcn','gacreationlinearfeasible','CrossoverFcn','crossoversinglepoint',...
'CrossoverFraction',0.5,'MutationFcn','mutationadaptfeasible','display',...
'iter','TolFun',1e-4,'PopulationSize',10,'MaxGenerations',10,...
'MaxStallGenerations',10);
%% GA run
[x0 , fval] = gamultiobj(FitnessFunction,3,[],[],[],[],lb,ub,[],options);
x0 =floor(x0);
%% results visualization
[score, i] = knee_pt(fval(:,2),fval(:,1),[]);
opts = statset('Display','final');
[idx, C, E] = kmeans(X,floor(x0(i,1)),'Distance',DFunc(floor(x0(i,2))),...
        'Replicates',floor(x0(i,3)),'Options',opts);
    
figure;
scatter3(X(:,1), X(:,2), X(:,3), 15, idx, 'filled');
hold on
scatter3(C(:,1), C(:,2), C(:,3), 300, 'r', 'kx', 'LineWidth',2);
title 'Cluster Assignments and Centroids'
hold off

%% ObjFunction
% By default, kmeans uses the squared Euclidean distance metric and the k-means++ algorithm
function ObjF = ObjectFnc(x)
global X C DFunc
ObjF = zeros(2,1);
[idx, C, E] = kmeans(X,floor(x(1)),'Distance',DFunc(floor(x(2))),...
        'Replicates',floor(x(3)));
s = silhouette(X,idx);
ObjF(1) = -mean(s);
ObjF(2) = sum(E);
end
