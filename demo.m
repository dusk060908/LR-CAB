clc;clear;
addpath(genpath(pwd));
addpath('./utils/');
addpath('./metrics/');

rng(2026)

M = 20;
cntTimes = 10;

poolSize = 100;
bcIdx = initialRandom(poolSize, M, cntTimes);

rawFolder  = "./datasets";
procFolder = "./cmpResult/";

procDirs = dir(fullfile(procFolder, "**", "*"));
procDirs = procDirs([procDirs.isdir]);
procDirNames = string({procDirs.name})';
procDirNames = procDirNames(procDirNames ~= "." & procDirNames ~= "..");
procNameSet  = unique(lower(strtrim(procDirNames)));

keep = lower(["Caltech20", ...
              "FCT", ...
              "IS", ...
              "ISOLET", ...
              "LR", ...
              "LS", ...
              "MF", ...
              "MNIST", ...
              "Semeion", ...
              "USPS", ...
              ]);
rawList = dir(fullfile(rawFolder,"**","*.mat"));
rawList = rawList(~[rawList.isdir]);
rawNames = lower(string(erase({rawList.name}, ".mat")));
rawList  = rawList(ismember(rawNames, keep));

% Model-3 specification: only lambda and delta_max are searched.
para_lambda = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1];
para_delta  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

for i = 1:numel(rawList)
    rawPath = fullfile(rawList(i).folder, rawList(i).name);
    [~, dataName, ~] = fileparts(rawList(i).name);
    rawKey = lower(strtrim(string(dataName)));

    if ismember(rawKey, procNameSet)
        fprintf("[SKIP] Subfolder already exists: %s\n", dataName);
        continue;
    end

    fprintf("[DO] LR-CAB: %s\n", dataName);
    load(rawPath);

    k = length(unique(gt));
    nL = length(para_lambda);
    nD = length(para_delta);

    mean_result1 = [];
    std_result1 = [];
    allTimes = cell(nL, nD);
    allResults = cell(nL, nD);
    allavg_time = cell(nL, nD);
    allstd_time = cell(nL, nD);
    allobjs = cell(nL, nD);
    allalphas = cell(nL, nD);
    allclusterings = cell(nL, nD);

    for lambda_idx = 1:nL
        for delta_idx = 1:nD
            lambda = para_lambda(lambda_idx);
            delta_max = para_delta(delta_idx);

            resultsAll = cell(1, cntTimes);
            times = zeros(1, cntTimes);
            objsAll = cell(1, cntTimes);
            alphasAll = cell(1, cntTimes);
            clusteringsAll = cell(1, cntTimes);
            measure1 = zeros(8, cntTimes);

            parfor idx = 1:cntTimes
                clusterings = members(:, bcIdx(idx, :));
                tStart = tic;
                [result1, obj, alpha] = main( ...
                    clusterings, M, k, lambda, delta_max);
                times(idx) = toc(tStart);
                resultsAll{idx} = result1;
                objsAll{idx} = obj;
                alphasAll{idx} = alpha;
                clusteringsAll{idx} = clusterings;
                measure1(:, idx) = computeMetrics(result1, gt);
            end

            avg_time = mean(times);
            std_time = std(times);
            allTimes{lambda_idx, delta_idx} = times;
            allResults{lambda_idx, delta_idx} = resultsAll;
            allavg_time{lambda_idx, delta_idx} = avg_time;
            allstd_time{lambda_idx, delta_idx} = std_time;
            allobjs{lambda_idx, delta_idx} = objsAll;
            allalphas{lambda_idx, delta_idx} = alphasAll;
            allclusterings{lambda_idx, delta_idx} = clusteringsAll;

            mean_measure1 = mean(measure1, 2)';
            std_measure1 = std(measure1, 0, 2)';
            mean_result1 = [mean_result1; [lambda, delta_max, mean_measure1]];
            std_result1 = [std_result1; [lambda, delta_max, std_measure1]];

            fprintf(['lambda:%.4g,delta_max:%.2f,NMI:%.4f,ARI:%.4f,F1:%.4f,' ...
                     'Purity:%.4f,ACC:%.4f,Kappa:%.4f,Precision:%.4f,Recall:%.4f\n'], ...
                    lambda, delta_max, mean_measure1);
        end
    end

    % Select the parameter pair by the product of the eight evaluation metrics.
    [~, Idx] = max(prod(mean_result1(:, end-7:end), 2));
    mean1 = mean_result1(Idx, :);
    std1 = std_result1(Idx, :);

    if ~exist(fullfile('.', 'cmpResult', dataName), 'dir')
        mkdir(fullfile('.', 'cmpResult', dataName));
    end
    if ~exist(fullfile('.', 'para', dataName), 'dir')
        mkdir(fullfile('.', 'para', dataName));
    end

    save(fullfile('.', 'cmpResult', dataName, 'mean1'), 'mean1');
    save(fullfile('.', 'cmpResult', dataName, 'std1'), 'std1');
    save(fullfile('.', 'para', dataName, 'mean_result1'), 'mean_result1');
    save(fullfile('.', 'para', dataName, 'std_result1'), 'std_result1');

    outDir = fullfile('.', 'results', dataName);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    outFile = fullfile(outDir, 'data_save1.mat');
    save(outFile, 'allTimes', 'allResults', 'allavg_time', 'allstd_time', ...
         'allobjs', 'allalphas', 'allclusterings', 'para_lambda', 'para_delta');
end