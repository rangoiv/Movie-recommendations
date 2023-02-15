def single_weight(a, b):
    return (2*a*b)/(a*a+b*b)

#calculate distance between two vectors (two users)
def calc_distance(A, B):
    n = len(A)
    m = len(B)
    p1 = 0
    p2 = 0
    sol = 0
    br = 0
    while (p1 < n and p2 < m):
        if A[p1][0] < B[p2][0]:
            p1 += 1
        elif A[p1][0] > B[p2][0]:
            p2 += 1
        else:
            br += 1
            sol += single_weight(A[p1][1], B[p2][1])
            p1 += 1
            p2 += 1

    ret = 0
    if br:
        ret = sol/br

    return ret

def all_pairwise_distances(A):
    n = len(A)
    dist = []
    for i in range(n):
        temp = []
        for j in range(n):
            temp.append(0)
        dist.append(temp)

    for i in range(n):
        for j in range(i+1,n):
            dist[i][j] = dist[j][i] = calc_distance(A[i],A[j])

    return dist
                 
def knn_algorithm(K, userId, movieId, A, dist):
    sorted_pairs = []
    for i in range(len(A)):
        if i == userId:
            continue
        sorted_pairs.append((dist[userId][i], i))
        
    sorted_pairs.sort(reverse=True)
    
    counter = 0
    rating = [0]*11
    for (i,j) in sorted_pairs:
        for (k,l) in A[j]:
            if k == movieId:
                rating[int(l*2//1)]+=i
                counter += 1
        if counter == K:
            break
                
    ret = 0
    for i in range(11):
        if rating[i] > rating[ret]:
            ret = i
            
    return ret/2
        
    
    
