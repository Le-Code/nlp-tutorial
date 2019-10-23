# gradient descent
if 1:
    weights = [0] * n
    alpha = 0.0001
    max_Iter = 50000
    for i in range(max_Iter):
        loss = 0
        d_weights = [0] * n
        for k in range(m):
            h = dot(input[k], weights)
            d_weights = [d_weights[j] + (label[k] - h) * input[k][j] for j in range(n)]
            loss += (label[k] - h) * (label[k] - h) / 2
        d_weights = [d_weights[k]/m for k in range(n)]
        weights = [weights[k] + alpha * d_weights[k] for k in range(n)]
        if i%10000 == 0:
            print "Iteration %d loss: %f"%(i, loss/m)
            print weights
