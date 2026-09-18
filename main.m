function [y_ind, obj, alpha, diagnostics] = main(clusterings, M, k, lambda, delta_max)
%MAIN LR-CAB model aligned with the model-3 specification.
%   Searchable hyperparameters: lambda and delta_max only.
%   gamma is fixed to 2 through the alpha^2 objective in solver.

    [~, baseClsSegs] = getAllSegs(clusterings);
    [~, n] = size(baseClsSegs);

    baseClsSegs = sparse(baseClsSegs);

    eachClusters = max(clusterings);
    A = cell(1, M);
    startIdx = 1;
    for i = 1:M
        endIdx = startIdx + eachClusters(i) - 1;
        A{i} = sparse(baseClsSegs(startIdx:endIdx, :)');
        startIdx = endIdx + 1;
    end

    % Initialize the hard consensus labels with k-means on the normalized base-indicator embedding.
    Y = zeros(n, k);
    fea = baseClsSegs' * diag(sum(baseClsSegs').^(-1/2));
    now = litekmeans(fea, k);
    idx = sub2ind([n, k], (1:n)', now);
    Y(idx) = 1;

    % Model-3 parameter-free local reliability; frozen before optimization.
    [Q, confidence] = computeLocalReliability(clusterings);

    [y_ind, obj, alpha, diagnostics] = solver( ...
        A, Y, Q, confidence, lambda, delta_max);
end