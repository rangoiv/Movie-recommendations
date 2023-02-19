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

def all_distances(A, userId):
    n = len(A)
    dist = []
    for i in range(n):
        if i == userId:
            dist.append(0)
        else:
            dist.append(calc_distance(A[i], A[userId]))
        
    return dist
                 
def knn_algorithm(K, userId, A, dist):
    rating = [0]*11
    rating_dict = {}
    counting_dict = {}
    for (i,j) in A[userId]:
        rating_dict[i] = rating
        counting_dict[i] = 0

    sorted_pairs = []
    for i in range(len(A)):
        if i == userId:
            continue
        sorted_pairs.append((dist[i], i))
        
    sorted_pairs.sort(reverse=True)
    for (i,j) in sorted_pairs:
        for (k,l) in A[j]:
            if k not in counting_dict:
                continue
            if counting_dict[k] == K:
                continue
            rating_dict[k][int(l*2)] += i
            counting_dict[k] += 1

    ret = []
    for (i,k) in A[userId]:
        ind = 0
        for j in range(11):
            if rating_dict[i][j] > rating_dict[i][ind]:
                ind = j
        ret.append((i, ind/2))
    return ret
        
    
    
