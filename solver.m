function [y_ind, obj, alpha, diagnostics] = solver(A, Y, Q, confidence, lambda, delta_max)
%SOLVER Direct hard-label solver for LR-CAB.
%
% Objective:
%   sum_r alpha_r^2 sum_k
%       y_k' (Dtilde_r-Wtilde_r) y_k / (y_k' Dtilde_r y_k + eps)
%   + lambda sum_k psi([|n_k/(n/K)-1|-delta_max*cbar_k]_+),
% where psi(v)=sqrt(1+v^2)-1.
%
% No additional regularization, normalization, or preprocessing is used.

    [n, k] = size(Y);
    M = numel(A);

    if size(Q, 1) ~= n || size(Q, 2) ~= M
        error('Q must be n-by-M.');
    end
    if numel(confidence) ~= n
        error('confidence must contain one value per sample.');
    end

    % Fixed gamma=2 => alpha_r^2. Initialize base-partition weights uniformly.
    alpha = ones(M, 1) / M;

    % Build the prescribed reliability-weighted normalized bipartite graphs.
    Btilde = cell(1, M);
    degree = cell(1, M);
    U = cell(1, M);              % U{r}=Y' * Btilde_r
    S = zeros(k, M);             % S(k,r)=||Btilde_r' y_k||^2
    G = zeros(k, M);             % G(k,r)=y_k' Dtilde_r y_k

    for r = 1:M
        baseSize = full(sum(A{r}, 1));
        invSqrtSize = baseSize.^(-1/2);
        Br = A{r} * spdiags(invSqrtSize', 0, numel(baseSize), numel(baseSize));
        Btilde{r} = spdiags(sqrt(Q(:, r)), 0, n, n) * Br;

        % Dtilde_r = diag(Wtilde_r * 1), Wtilde_r=Btilde_r*Btilde_r'.
        colSum = full(sum(Btilde{r}, 1))';
        degree{r} = Btilde{r} * colSum;

        U{r} = Y' * Btilde{r};
        S(:, r) = full(sum(U{r}.^2, 2));
        G(:, r) = full(Y' * degree{r});
    end

    y1 = sum(Y, 1)';
    y_ind = vec2ind(Y')';
    C = Y' * confidence(:);      % C_k=sum_{i in cluster k} c_i

    maxOuter = 10;               % Fixed outer iteration budget.
    obj = zeros(1, maxOuter);
    innerHistory = cell(1, maxOuter);

    for iter = 1:maxOuter
        [y_ind, innerObj, y1, C, Y, U, S, G] = update_Y_LRCAB( ...
            Btilde, degree, alpha, lambda, delta_max, confidence, ...
            y_ind, y1, C, Y, U, S, G);
        innerHistory{iter} = innerObj;

        % Closed-form alpha update for fixed gamma=2.
        E = (G - S) ./ (G + eps);
        Er = sum(E, 1)';
        invE = 1 ./ (Er + eps);
        alpha = invE ./ sum(invE);

        % Exact fixed-objective value after both blocks are updated.
        obj(iter) = consensus_objective(Er, alpha) + ...
                    balance_objective(y1, C, n, k, lambda, delta_max);

        if iter > 1 && abs(obj(iter) - obj(iter - 1)) < 1e-5
            obj = obj(1:iter);
            innerHistory = innerHistory(1:iter);
            break;
        end
    end

    diagnostics.Q = Q;
    diagnostics.confidence = confidence(:);
    diagnostics.cluster_sizes = y1;
    diagnostics.cluster_confidence_sum = C;
    diagnostics.inner_objective = innerHistory;
end

function [y_ind, obj, y1, C, Y, U, S, G] = update_Y_LRCAB( ...
    Btilde, degree, alpha, lambda, delta_max, confidence, ...
    y_ind, y1, C, Y, U, S, G)

    n = size(Y, 1);
    k = size(Y, 2);
    M = numel(Btilde);
    maxInner = 10;               % Fixed coordinate-sweep budget.
    obj = zeros(1, maxInner + 1);

    for sweep = 1:maxInner
        E = (G - S) ./ (G + eps);
        Er = sum(E, 1)';
        obj(sweep) = consensus_objective(Er, alpha) + ...
                     balance_objective(y1, C, n, k, lambda, delta_max);

        moveCount = 0;
        for ii = 1:n
            p = y_ind(ii);

            % Keep every consensus cluster non-empty.
            if y1(p) == 1
                continue;
            end

            oldE = (G - S) ./ (G + eps);
            delta = zeros(k, 1);
            tCache = zeros(k, M);
            b2Cache = zeros(M, 1);
            dCache = zeros(M, 1);

            % Exact incremental change of the LR normalized-cut term.
            for r = 1:M
                br = Btilde{r}(ii, :);
                t = full(U{r} * br');
                b2 = full(br * br');
                di = full(degree{r}(ii));

                tCache(:, r) = t;
                b2Cache(r) = b2;
                dCache(r) = di;

                gp = G(p, r) - di;
                sp = S(p, r) - 2 * t(p) + b2;
                epAfter = (gp - sp) / (gp + eps);

                gqAfter = G(:, r) + di;
                sqAfter = S(:, r) + 2 * t + b2;
                eqAfter = (gqAfter - sqAfter) ./ (gqAfter + eps);

                delta = delta + alpha(r)^2 .* ...
                    ((epAfter - oldE(p, r)) + (eqAfter - oldE(:, r)));
            end

            % Exact incremental change of confidence-adaptive balancing.
            psiOld = balance_psi_vector(y1, C, n, k, delta_max);

            npAfter = y1(p) - 1;
            CpAfter = C(p) - confidence(ii);
            cbarP = CpAfter / npAfter;
            vp = max(abs(npAfter / (n / k) - 1) - delta_max * cbarP, 0);
            psiPAfter = sqrt(1 + vp^2) - 1;

            nqAfter = y1 + 1;
            CqAfter = C + confidence(ii);
            cbarQ = CqAfter ./ nqAfter;
            vq = max(abs(nqAfter / (n / k) - 1) - delta_max .* cbarQ, 0);
            psiQAfter = sqrt(1 + vq.^2) - 1;

            delta = delta + lambda .* ...
                ((psiPAfter - psiOld(p)) + (psiQAfter - psiOld));

            % q=p means no move; its exact change is zero.
            delta(p) = 0;
            [bestDelta, q] = min(delta);

            if q ~= p && bestDelta < 0
                for r = 1:M
                    br = Btilde{r}(ii, :);
                    t = tCache(:, r);
                    b2 = b2Cache(r);
                    di = dCache(r);

                    S(p, r) = S(p, r) - 2 * t(p) + b2;
                    S(q, r) = S(q, r) + 2 * t(q) + b2;
                    G(p, r) = G(p, r) - di;
                    G(q, r) = G(q, r) + di;

                    U{r}(p, :) = U{r}(p, :) - br;
                    U{r}(q, :) = U{r}(q, :) + br;
                end

                y1([p, q]) = y1([p, q]) + [-1; 1];
                C([p, q]) = C([p, q]) + [-confidence(ii); confidence(ii)];
                y_ind(ii) = q;
                Y(ii, p) = 0;
                Y(ii, q) = 1;
                moveCount = moveCount + 1;
            end
        end

        E = (G - S) ./ (G + eps);
        Er = sum(E, 1)';
        obj(sweep + 1) = consensus_objective(Er, alpha) + ...
                         balance_objective(y1, C, n, k, lambda, delta_max);

        if moveCount == 0 || abs(obj(sweep + 1) - obj(sweep)) < 1e-5
            obj = obj(1:sweep + 1);
            break;
        end
    end
end

function val = consensus_objective(Er, alpha)
    val = sum((alpha.^2) .* Er);
end

function val = balance_objective(y1, C, n, k, lambda, delta_max)
    psi = balance_psi_vector(y1, C, n, k, delta_max);
    val = lambda * sum(psi);
end

function psi = balance_psi_vector(y1, C, n, k, delta_max)
    cbar = C ./ y1;
    v = max(abs(y1 / (n / k) - 1) - delta_max .* cbar, 0);
    psi = sqrt(1 + v.^2) - 1;
end