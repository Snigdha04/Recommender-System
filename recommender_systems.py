"""
Created on 15th October, 2018

@author: Aditya Desai
         Snigdha GRandhi
         Aayush Barathwal
         Meghna Vasudeva
"""
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.linalg import norm
from numpy import dot
import math

def read_data(filepath):
    """
    This function reads the contents of the respective csv file which is the parameter which is passed.
    It returns the utility matrix which is initialised with the corresponding ratings of the user and movie.
    """
    data = pd.read_csv(filepath,header=None,delimiter='\t',names=["userId","movieId","rating","timestamp"])
    utility = np.zeros((max(data["movieId"])+1,max(data["userId"])+1))
    for i in range(len(data)):
        utility[data["movieId"][i],data["userId"][i]] = data["rating"][i]
    return utility

def split(x):
    """
    This function is used for randomly splitting the dataset into training(70% of entire dataset) and test set(30% of entire dataset).
    It is passed the original dataset and it returns:
    ->the training set which now holds 70% of the original values
    ->test set which holds 30% of the values which are randomly selected
    ->coordinates of the randomly selected values.
    """
    num = len(x) if len(x)>len(x[0]) else len(x[0])
    test = np.random.randint(num, size=((int)(0.3*x.size), 2))
    test[:, 0] = test[:, 0]%len(x)
    test[:, 1] = test[:, 1]%len(x[0])
    xactual = [x[xx[0]][xx[1]] for xx in test]
    for val in test:
        x[val[0]][val[1]] = 0
    xactual = np.array(xactual)
    return x, xactual, test

def normalize(utility):
    # normalizing the utility matrix
    normalized_utility = np.zeros( (utility.shape[0],utility.shape[1]) )
    for i in range(1,utility.shape[0]):
        num = np.count_nonzero(utility[i])
        if num!=0: row_mean = sum(utility[i]) / num
        else: row_mean = 0
        for j in range(1,utility.shape[1]):
            if utility[i,j]!=0:
                normalized_utility[i,j] = utility[i,j] - row_mean
    return normalized_utility

def cfpred(utility, sim_matrix,movieid,userid,nearestn=3):
    temp = []
    for i in range(len(utility)):
        if i!=movieid and utility[i,userid]!=0 :
            temp.append(  (i,sim_matrix[movieid,i]) )
    temp = sorted(temp, key=lambda x:x[1],reverse=True)
    temp = temp[:nearestn]
    # print(temp)
    numerator = 0
    denominator = 0
    for el in temp:
        numerator+= utility[el[0],userid]*el[1]
        denominator += el[1]
    return numerator/denominator

def precitionatK(utility, normalized_utility, xtest, sim_matrix):
    k = 5
    average_pak = 0
    counter=0
    for user in range(1,utility.shape[1]):
        actual = []
        predicted = []
        for movie in range(1,utility.shape[0]):
            if normalized_utility[movie,user]!=0:
                actual.append((movie,normalized_utility[movie,user]))
                predicted.append((movie,cfpred(xtest.T,sim_matrix,movieid=movie,userid=user,nearestn=1)))
        actual = sorted(actual,key=lambda x:x[1],reverse=True)
        predicted = sorted(predicted,key=lambda x:x[1], reverse=True)
        actual=actual[:k]
        predicted=predicted[:k]

        actual = [x[0] for x in actual]
        predicted = [x[0] for x in predicted]
        pak = 0
        for el in actual:
            if el in predicted:
                pak+=1
        pak/=k
        average_pak+=pak
        counter+=1
    average_pak/=counter
    print('precition at k : ')
    print(average_pak)

def find_top(a):
    # This function returns the indices of the top 3 similarites after sorting in decreasing order.
    temp = np.argsort(-a)
    return temp[1:4]

def rmse(predicted, xactual):
    # This function prints the root mean square error between the test set(30% of dataset) and the same values post recommending.
    err = np.sqrt(np.sum((xactual - predicted)**2)/len(xactual))
    print('rmse : ')
    print(err)

def spearman(predicted, xactual):
    # This function prints the spearman rank correlation  between the test set(30% of dataset) and the same values post recommending.n
    err = np.sum(6*(xactual - predicted)**2)
    err = 1 - (err)/(len(xactual)*(len(xactual)**2 - 1))
    print('spearman rank correlation : ')
    print(err)

def cosine_similarity(xtest, item_avgs):
    """
    This function returns the cosine centered similarity of the item(movie) columns.
    It's parameters are the training data set and the itme averages which are subtracted from each value before computing the dot product.
    """
    # subtracting the item average
    for i in range(len(xtest[0])):
        for j in range(len(xtest)):
            if xtest[j][i]!=0 : xtest[j][i] -= item_avgs[i]

    # finding the cosine similarity
    similarity = []
    result_data_trans = xtest.T
    for i in range(len(result_data_trans)):
        temp = []
        for j in range(len(result_data_trans)):
            a = result_data_trans[i]
            b = result_data_trans[j]
            if norm(a)!=0 and norm(b)!=0:
                sim = dot(a, b)/(norm(a)*norm(b))
            else : sim=0
            temp.append(sim)
        temp = np.array(temp)
        similarity.append(temp)
    similarity = np.array(similarity)
    return similarity

def averages(xtest):
    # This function is passed the training set and it returns the item average, user average and entire dataset average
    item_avgs = []
    user_avgs = []
    mat_avg = 0
    # computing the user average and matrix average
    for i in range(len(xtest)):
        sumval = 0
        cnt = 0
        for j in range(len(xtest[0])):
            if xtest[i][j]!=0:
                sumval += xtest[i][j]
                cnt += 1
                mat_avg += xtest[i][j]
        if cnt!=0: avg = sumval/cnt
        else : avg = 0
        user_avgs.append(avg)
    mat_avg = mat_avg/(len(xtest)*len(xtest[0]))

    # computing the item average
    for i in range(len(xtest[0])):
        sumval = 0
        cnt = 0
        for j in range(len(xtest)):
            if xtest[j][i]!=0:
                sumval += xtest[j][i]
                cnt += 1
        if cnt!=0 : avg = sumval/cnt
        else : avg = 0
        item_avgs.append(avg)
    return item_avgs, user_avgs, mat_avg

def collaborative_recommender(xtest, coordinates, xtestsim):
    """
    This function does collaborative filtering. It's parameters are the training set and the coordinates of the values in the test set.
    It first calls the averages function, it uses the item avaerages from there and passes on to cosine_similarity funtion.
    The cosine_similarity funtion stores the similarities in the similarity matrix.
    Then for each item, it finds the top few similar similarities by passing the column in the similarity matrix to the find_top function.
    Then it takes the weighted average of the top few similarites and it's ratings and stores them in a list called predicted which it returns.
    """
    item_avgs, user_avgs, mat_avg = averages(xtestsim)



    similarity = cosine_similarity(xtestsim, item_avgs)
    # predicting the unknown values
    start = time.time()
    predicted = []
    for i in coordinates:
        col = i[1]
        # find similarity using top 10vals
        topten_index = find_top(similarity[col])
        numerator = 0
        denominator = 0
        # collaborative
        for ii in topten_index:
            numerator += similarity[col][ii]*xtest[i[0]][ii]
            denominator += similarity[col][ii]
        if denominator!=0: predicted.append(numerator/denominator)
        else: predicted.append(0)
    predicted = np.array(predicted)
    end = time.time()

    return predicted, (end-start)/len(predicted), similarity

def collaborative_recommender_with_baseline(xtest, coordinates):
    """
    This funtion does collaborative filtering along with baseline. It's parameters are the training set and the coordinates of the values in the test set.
    It is similar to the collaborative_recommender function, it only differs from it whilst computing the weighted averages.
    It subtracts the baseline estimate of the respective user and item from each rating when taking the weighted average and in the end, adds the baseline estimate
    for the respective user and item.
    """
    item_avgs, user_avgs, mat_avg = averages(xtest)
#

    similarity = cosine_similarity(xtest, item_avgs)


    # predicting the unknown values
    start = time.time()
    predicted = []
    for i in coordinates:
        col = i[1]
        # find similarity using top 10 vals
        topten_index = find_top(similarity[col])
        numerator = 0
        denominator = 0
        for ii in topten_index:
            bxj = mat_avg + user_avgs[i[0]] + item_avgs[ii]
            numerator += similarity[col][ii]*(xtest[i[0]][ii] - (bxj))
            denominator += similarity[col][ii]
        bxi = mat_avg + user_avgs[i[0]] + item_avgs[i[1]]
        if denominator!=0: val = bxi+(numerator/denominator)
        else: val = bxi
        predicted.append(val)
    predicted = np.array(predicted)
    end = time.time()
    return predicted , (end-start)/len(predicted)

def svd(xtest, coordinates):
    """
    This function  does the SVD technique.
    It's parameters are the training set and the coordinates of the values in the test set.

    """
    e_u,u = np.linalg.eig(np.dot(xtest , xtest.transpose()))
    e_v,v = np.linalg.eig(np.dot(xtest.transpose() , xtest))
    vh = v.T
    return collaborative_recommender(xtest, coordinates, np.dot(xtest,vh))

def num_eig_for_90(e_v):

    total = np.sum(e_v)
    percentage = 0
    counter = 0
    while((percentage/total)<=0.9):
        percentage += e_v[counter]
        counter+=1
    return counter

def svd90(xtest, coordinates):
    """
    This function  does the SVD technique with 90% retention.
    It's parameters are the training set and the coordinates of the values in the test set.

    """
    e_u,u = np.linalg.eig(np.dot(xtest , xtest.transpose()))
    e_v,v = np.linalg.eig(np.dot(xtest.transpose() , xtest))
    vh = v.transpose()

    num = num_eig_for_90(e_v)
    u = u[:, :num]
    vh = vh[:num, :]
    pred,timetaken = collaborative_recommender(xtest, coordinates, np.dot(xtest,vh))
    return pred, timetaken

def cur(xtest, coordinates):
    # find the probabilty vectors for rows and cols
    prow = np.sum(xtest**2,axis=1)/ np.sum(np.sum(xtest**2,axis=1))
    pcol = np.sum(xtest**2,axis=0)/ np.sum(np.sum(xtest**2,axis=1))

    # select k entries of both rows and cols
    k=1000
    rowidx = np.random.choice(len(prow),k,p=prow)
    colidx = np.random.choice(len(pcol),k,p=pcol)
    C = np.copy(xtest[:,colidx])
    R = np.copy(xtest[rowidx,:])
    for i in range(k):
        C[:,i]/=(k*pcol[colidx[i]])**0.5
        R[i,:]/=(k*prow[rowidx[i]])**0.5
#     W = np.zeros((k,k))
#     for i in range(len(rowidx)):
#         for j in range(len(colidx)):
#             W[i,j] = xtest[rowidx[i], colidx[j]]
#     X, Z, Yt = np.linalg.svd(W, full_matrices=False)

#     Zmat = np.zeros( (X.shape[0],Yt.shape[0]) , dtype=complex )
#     Zmat[:X.shape[0],:X.shape[0]] = np.diag(Z)

#     for i in range(Zmat.shape[0]):
#         for j in range(Zmat.shape[1]):
#             if Zmat[i,j]!=0:
#                 Zmat[i,j] = 1/Zmat[i,j]
#     U = np.dot(np.dot(np.dot(Yt.transpose(), Zmat), Zmat), X.transpose())
    return collaborative_recommender(xtest, coordinates, R)


def cur90(xtest, coordinates):
    # find the probabilty vectors for rows and cols
    prow = np.sum(xtest**2,axis=1)/ np.sum(np.sum(xtest**2,axis=1))
    pcol = np.sum(xtest**2,axis=0)/ np.sum(np.sum(xtest**2,axis=1))

    # select k entries of both rows and cols
    k=900
    rowidx = np.random.choice(len(prow),k,p=prow)
    colidx = np.random.choice(len(pcol),k,p=pcol)
    C = np.copy(xtest[:,colidx])
    R = np.copy(xtest[rowidx,:])
    for i in range(k):
        C[:,i]/=(k*pcol[colidx[i]])**0.5
        R[i,:]/=(k*prow[rowidx[i]])**0.5
#     W = np.zeros((k,k))
#     for i in range(len(rowidx)):
#         for j in range(len(colidx)):
#             W[i,j] = xtest[rowidx[i], colidx[j]]
#     X, Z, Yt = np.linalg.svd(W, full_matrices=False)

#     Zmat = np.zeros( (X.shape[0],Yt.shape[0]) , dtype=complex )
#     Zmat[:X.shape[0],:X.shape[0]] = np.diag(Z)

#     for i in range(Zmat.shape[0]):
#         for j in range(Zmat.shape[1]):
#             if Zmat[i,j]!=0:
#                 Zmat[i,j] = 1/Zmat[i,j]
#     U = np.dot(np.dot(np.dot(Yt.transpose(), Zmat), Zmat), X.transpose())
    return collaborative_recommender(xtest, coordinates,R)

def main():
    # read the data
    np.random.seed(1)
    utility = read_data("./ml-100k/u.csv")
    result_data = utility.T

    # split the data into training and testing
    xtest, xactual, coordinates = split(result_data)

    predcr,timetaken,sim_matrix = collaborative_recommender(xtest, coordinates, xtest)
    rmse(predcr, xactual)
    precitionatK(utility, normalize(utility), xtest, sim_matrix)
    spearman(predcr, xactual)
    print("time taken per prediction for cf : " + str(timetaken))

    predcrwb,timetaken = collaborative_recommender_with_baseline(xtest, coordinates)
    rmse(predcrwb, xactual)
    precitionatK(utility, normalize(utility), xtest, sim_matrix)
    spearman(predcrwb, xactual)
    print("time taken per prediction for cf with baseline: " + str(timetaken))

    predsvd,timetaken,sim_matrix = svd(xtest, coordinates)
    rmse(predsvd, xactual)
    precitionatK(utility, normalize(utility), xtest, sim_matrix)
    spearman(predsvd, xactual)
    print("time taken per prediction for svd : " + str(timetaken))

    predsvd90,timetaken = svd90(xtest, coordinates)
    rmse(predsvd90, xactual)
    precitionatK(utility, normalize(utility), xtest, sim_matrix)
    spearman(predsvd90, xactual)
    print("time taken per prediction for svd with 90% energy retained : " + str(timetaken))

    predcur,timetaken,sim_matrix = cur(xtest, coordinates)
    rmse(predcur, xactual)
    precitionatK(utility, normalize(utility), xtest, sim_matrix)
    spearman(predcur, xactual)
    print("time taken per prediction for cur : " + str(timetaken))

    predcur90,timetaken = cur90(xtest, coordinates)
    rmse(predcur90, xactual)
    precitionatK(utility, normalize(utility), xtest, sim_matrix)
    spearman(predcur90, xactual)
    print("time taken per prediction for cur  with 90% energy retained : " + str(timetaken))



if __name__ == "__main__":
	main()
