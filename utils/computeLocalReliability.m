function [Q, confidence] = computeLocalReliability(clusterings)
%COMPUTELOCALRELIABILITY Parameter-free sample-partition reliability for LR-CAB.
%   Q(i,r) = 1/(M-1) * sum_{s~=r} |C_r(i) cap C_s(i)| /
%            sqrt(|C_r(i)| |C_s(i)|).
%
%   This function uses only the ensemble labels. It does not use features,
%   ground-truth labels, extra normalization, or preprocessing.

    [n, M] = size(clusterings);
    if M < 2
        error('LR-CAB requires at least two base clusterings.');
    end

    Q = zeros(n, M);

    clusterSizes = cell(1, M);
    numClusters = zeros(1, M);
    for r = 1:M
        z = clusterings(:, r);
        if any(z < 1) || any(z ~= floor(z))
            error('Base-clustering labels must be positive integer indices, as required by getAllSegs.');
        end
        numClusters(r) = max(z);
        clusterSizes{r} = accumarray(z, 1, [numClusters(r), 1]);
    end

    for r = 1:M-1
        zr = clusterings(:, r);
        nr = clusterSizes{r};
        for s = r+1:M
            zs = clusterings(:, s);
            ns = clusterSizes{s};

            % Pairwise contingency table. For every sample i, the indexed
            % entry is exactly |C_r(i) cap C_s(i)|.
            Nrs = sparse(zr, zs, 1, numClusters(r), numClusters(s));
            linearIdx = sub2ind([numClusters(r), numClusters(s)], zr, zs);
            inter = full(Nrs(linearIdx));

            overlap = inter ./ sqrt(nr(zr) .* ns(zs));
            Q(:, r) = Q(:, r) + overlap;
            Q(:, s) = Q(:, s) + overlap;
        end
    end

    Q = Q ./ (M - 1);
    confidence = mean(Q, 2);
end