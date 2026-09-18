function [label, center, bCon, sumD, D] = litekmeans(X, k, varargin)
%LITEKMEANS K-means clustering, accelerated by matlab matrix operations.
%
%   label = LITEKMEANS(X, K) partitions the points in the N-by-P data matrix
%   X into K clusters.  This partition minimizes the sum, over all
%   clusters, of the within-cluster sums of point-to-cluster-centroid
%   distances.  Rows of X correspond to points, columns correspond to
%   variables.  KMEANS returns an N-by-1 vector label containing the
%   cluster indices of each point.
%
%   [label, center] = LITEKMEANS(X, K) returns the K cluster centroid
%   locations in the K-by-P matrix center.
%
%   [label, center, bCon] = LITEKMEANS(X, K) returns the bool value bCon to
%   indicate whether the iteration is converged.
%
%   [label, center, bCon, SUMD] = LITEKMEANS(X, K) returns the
%   within-cluster sums of point-to-centroid distances in the 1-by-K vector
%   sumD.
%
%   [label, center, bCon, SUMD, D] = LITEKMEANS(X, K) returns
%   distances from each point to every centroid in the N-by-K matrix D.
%
%   NOTE: The original implementation contained a hard-coded `rng(2026);`
%   inside the function body. That has been removed so that the random
%   stream is fully controlled by the caller (e.g. demo.m). This is
%   required for reproducibility inside parfor, where multiple workers
%   would otherwise repeatedly reset the global stream.

if nargin < 2
    error('litekmeans:TooFewInputs','At least two input arguments required.');
end

[n, p] = size(X);

pnames = {   'distance' 'start'   'maxiter'  'replicates' 'onlinephase' 'clustermaxiter'};
dflts =  {'sqeuclidean' 'sample'       []        []        'off'              []        };
[eid,errmsg,distance,start,maxit,reps,online,clustermaxit] = getargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('litekmeans:%s',eid),errmsg);
end

if ischar(distance)
    distNames = {'sqeuclidean','cosine'};
    j = strcmpi(distance, distNames);
    j = find(j);
    if length(j) > 1
        error('litekmeans:AmbiguousDistance', ...
            'Ambiguous ''Distance'' parameter value:  %s.', distance);
    elseif isempty(j)
        error('litekmeans:UnknownDistance', ...
            'Unknown ''Distance'' parameter value:  %s.', distance);
    end
    distance = distNames{j};
else
    error('litekmeans:InvalidDistance', ...
        'The ''Distance'' parameter value must be a string.');
end

center = [];
if ischar(start)
    startNames = {'sample','cluster'};
    j = find(strncmpi(start,startNames,length(start)));
    if length(j) > 1
        error(message('litekmeans:AmbiguousStart', start));
    elseif isempty(j)
        error(message('litekmeans:UnknownStart', start));
    elseif isempty(k)
        error('litekmeans:MissingK', ...
            'You must specify the number of clusters, K.');
    end
    if j == 2
        if floor(.1*n) < 5*k
            j = 1;
        end
    end
    start = startNames{j};
elseif isnumeric(start)
    if size(start,2) == p
        center = start;
    elseif (size(start,2) == 1 || size(start,1) == 1)
        center = X(start,:);
    else
        error('litekmeans:MisshapedStart', ...
            'The ''Start'' matrix must have the same number of columns as X.');
    end
    if isempty(k)
        k = size(center,1);
    elseif (k ~= size(center,1))
        error('litekmeans:MisshapedStart', ...
            'The ''Start'' matrix must have K rows.');
    end
    start = 'numeric';
else
    error('litekmeans:InvalidStart', ...
        'The ''Start'' parameter value must be a string or a numeric matrix or array.');
end

if isempty(maxit)
    maxit = 100;
end

if isempty(clustermaxit)
    clustermaxit = 10;
end

if isempty(reps) || ~isempty(center)
    reps = 1;
end

if ~(isscalar(k) && isnumeric(k) && isreal(k) && k > 0 && (round(k)==k))
    error('litekmeans:InvalidK', ...
        'X must be a positive integer value.');
elseif n < k
    error('litekmeans:TooManyClusters', ...
        'X must have more rows than the number of clusters.');
end

bestlabel = [];
sumD = zeros(1,k);
bCon = false;

for t=1:reps
    switch start
        case 'sample'
            center = X(randsample(n,k),:);
        case 'cluster'
            Xsubset = X(randsample(n,floor(.1*n)),:);
            [dump, center] = litekmeans(Xsubset, k, varargin{:}, 'start','sample', 'replicates',1 ,'MaxIter',clustermaxit);
        case 'numeric'
    end

    last = 0;label=1;
    it=0;

    switch distance
        case 'sqeuclidean'
            while any(label ~= last) && it<maxit
                last = label;

                bb = full(sum(center.*center,2)');
                ab = full(X*center');
                D = bb(ones(1,n),:) - 2*ab;

                [val,label] = min(D,[],2);
                ll = unique(label);
                if length(ll) < k
                    missCluster = 1:k;
                    missCluster(ll) = [];
                    missNum = length(missCluster);

                    aa = sum(X.*X,2);
                    val = aa + val;
                    [dump,idx] = sort(val,1,'descend');
                    label(idx(1:missNum)) = missCluster;
                end
                E = sparse(1:n,label,1,n,k,n);
                center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);
                it=it+1;
            end
            if it<maxit
                bCon = true;
            end
            if isempty(bestlabel)
                bestlabel = label;
                bestcenter = center;
                if reps>1
                    if it>=maxit
                        aa = full(sum(X.*X,2));
                        bb = full(sum(center.*center,2));
                        ab = full(X*center');
                        D = bsxfun(@plus,aa,bb') - 2*ab;
                        D(D<0) = 0;
                    else
                        aa = full(sum(X.*X,2));
                        D = aa(:,ones(1,k)) + D;
                        D(D<0) = 0;
                    end
                    D = sqrt(D);
                    for j = 1:k
                        sumD(j) = sum(D(label==j,j));
                    end
                    bestsumD = sumD;
                    bestD = D;
                end
            else
                if it>=maxit
                    aa = full(sum(X.*X,2));
                    bb = full(sum(center.*center,2));
                    ab = full(X*center');
                    D = bsxfun(@plus,aa,bb') - 2*ab;
                    D(D<0) = 0;
                else
                    aa = full(sum(X.*X,2));
                    D = aa(:,ones(1,k)) + D;
                    D(D<0) = 0;
                end
                D = sqrt(D);
                for j = 1:k
                    sumD(j) = sum(D(label==j,j));
                end
                if sum(sumD) < sum(bestsumD)
                    bestlabel = label;
                    bestcenter = center;
                    bestsumD = sumD;
                    bestD = D;
                end
            end
        case 'cosine'
            while any(label ~= last) && it<maxit
                last = label;
                W=full(X*center');
                [val,label] = max(W,[],2);
                ll = unique(label);
                if length(ll) < k
                    missCluster = 1:k;
                    missCluster(ll) = [];
                    missNum = length(missCluster);
                    [dump,idx] = sort(val);
                    label(idx(1:missNum)) = missCluster;
                end
                E = sparse(1:n,label,1,n,k,n);
                center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);
                centernorm = sqrt(sum(center.^2, 2));
                center = center ./ centernorm(:,ones(1,p));
                it=it+1;
            end
            if it<maxit
                bCon = true;
            end
            if isempty(bestlabel)
                bestlabel = label;
                bestcenter = center;
                if reps>1
                    if any(label ~= last)
                        W=full(X*center');
                    end
                    D = 1-W;
                    for j = 1:k
                        sumD(j) = sum(D(label==j,j));
                    end
                    bestsumD = sumD;
                    bestD = D;
                end
            else
                if any(label ~= last)
                    W=full(X*center');
                end
                D = 1-W;
                for j = 1:k
                    sumD(j) = sum(D(label==j,j));
                end
                if sum(sumD) < sum(bestsumD)
                    bestlabel = label;
                    bestcenter = center;
                    bestsumD = sumD;
                    bestD = D;
                end
            end
    end
end

label = bestlabel;
center = bestcenter;
if reps>1
    sumD = bestsumD;
    D = bestD;
elseif nargout > 3
    switch distance
        case 'sqeuclidean'
            if it>=maxit
                aa = full(sum(X.*X,2));
                bb = full(sum(center.*center,2));
                ab = full(X*center');
                D = bsxfun(@plus,aa,bb') - 2*ab;
                D(D<0) = 0;
            else
                aa = full(sum(X.*X,2));
                D = aa(:,ones(1,k)) + D;
                D(D<0) = 0;
            end
            D = sqrt(D);
        case 'cosine'
            if it>=maxit
                W=full(X*center');
            end
            D = 1-W;
    end
    for j = 1:k
        sumD(j) = sum(D(label==j,j));
    end
end


function [eid,emsg,varargout]=getargs(pnames,dflts,varargin)
%GETARGS Process parameter name/value pairs
emsg = '';
eid = '';
nparams = length(pnames);
varargout = dflts;
unrecog = {};
nargs = length(varargin);

if mod(nargs,2)~=0
    eid = 'WrongNumberArgs';
    emsg = 'Wrong number of arguments.';
else
    for j=1:2:nargs
        pname = varargin{j};
        if ~ischar(pname)
            eid = 'BadParamName';
            emsg = 'Parameter name must be text.';
            break;
        end
        i = strcmpi(pname,pnames);
        i = find(i);
        if isempty(i)
            if nargout > nparams+2
                unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};
            else
                eid = 'BadParamName';
                emsg = sprintf('Invalid parameter name:  %s.',pname);
                break;
            end
        elseif length(i)>1
            eid = 'BadParamName';
            emsg = sprintf('Ambiguous parameter name:  %s.',pname);
            break;
        else
            varargout{i} = varargin{j+1};
        end
    end
end

varargout{nparams+1} = unrecog;