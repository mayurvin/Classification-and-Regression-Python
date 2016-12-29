import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
	classVector = np.unique(y)
	tempMeans = []
	correctedMean = []
	meanForX = np.mean(X,axis=0)
	covmatTemp = []
	for eachClass in classVector:
		Xg = X[y.flatten() == eachClass, :]
		tempMeans.append(Xg.mean(0))
		correctedMean.append(Xg - meanForX)
	
	means = np.transpose(np.asarray(tempMeans))
	covmat = np.cov(X,rowvar=0)
	return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
	classVector = np.unique(y)
	tempMeans = []
	correctedMean = []
	covmatTemp = []
	for eachClass in classVector:
		Xg = X[y.flatten() == eachClass, :]
		tempMeans.append(Xg.mean(0))
				
	covmats = []
	meanForX = np.mean(X,axis=0)
	for eachClass in classVector:
		Xg = X[y.flatten() == eachClass, :]
		# Xg = Xg - meanForX
		covmats.append(np.cov(Xg, rowvar =0))
    
	return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
	D = len(means)
	yPredicted = None
	means = means.transpose()
	for row in Xtest:
		listOfPDF = []
		for mean in means:
			xMinusMean = np.subtract(row,mean)
			invCovmat = np.linalg.inv(covmat)
			detCovmat = np.linalg.det(covmat)
			numeratorOfExp = (-1/2) * np.dot(np.dot(xMinusMean, invCovmat), np.transpose(xMinusMean))
			significant = 1 / (np.power((2 * 3.142), D/2) * np.power(detCovmat,0.5))
			pdf = float(significant * np.exp(numeratorOfExp))
			listOfPDF.append(pdf)
		maxPDF = max(listOfPDF)
		if yPredicted is None:
			yPredicted = np.array(float(listOfPDF.index(maxPDF) + 1))
		else:
			yPredicted = np.vstack((yPredicted, float(listOfPDF.index(maxPDF) + 1)))
		# acc = str(100*np.mean((yPredicted == ytest).astype(float)))
	acc = str(100*np.mean((yPredicted == ytest).astype(float)))
	
	return acc,yPredicted

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
	D = len(means)
	yPredicted = None
	means = means.transpose()
	for row in Xtest:
		listOfPDF = []
		for index in range(len(means)):            
			mean = means[index]   
			covmat = covmats[index]
            
			xMinusMean = np.subtract(row,mean)
			invCovmat = np.linalg.inv(covmat)
			detCovmat = np.linalg.det(covmat)
			numeratorOfExp = (-1/2) * np.dot(np.dot(xMinusMean, invCovmat), np.transpose(xMinusMean))
			significant = 1 / (np.power((2 * 3.142), D/2) * np.power(detCovmat,0.5))
			pdf = float(significant * np.exp(numeratorOfExp))
			listOfPDF.append(pdf)
		maxPDF = max(listOfPDF)
        
		if yPredicted is None:
			yPredicted = np.array(float(listOfPDF.index(maxPDF) + 1))
		else:
			yPredicted = np.vstack((yPredicted, float(listOfPDF.index(maxPDF) + 1)))
	acc = 100*np.mean((yPredicted == ytest).astype(float))
	
	return acc,yPredicted

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD             
    w = np.dot(inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
	#Formula for w = inverse[(XT X + Lambda*I)].XT y
	
	#Calculating Lambda*I
	#print X.shape
	I = np.identity(X.shape[1])
	lambdaI = lambd*I
	
	#Calculating XT X
	Xt = np.transpose(X)
	XtX = np.dot(Xt,X)
	
	inverseTerm = XtX + lambdaI
	#taking inverse of inverseTerm
	weightPart1 = np.linalg.inv(inverseTerm)
	
	#Calculating XT Y
	XtY = np.dot(Xt,y)
	
	#Calculating final weight
	w = np.dot(weightPart1, XtY)
	return w                                                  

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
	
	#Calculating X*w
	a = np.dot(Xtest,w)
	
	#Calculating y-X*w
	b = ytest - a
	
	#Calculating (y-X*w)Transpose
	c = np.transpose(b)
	
	#Calculating (y-X*w)Transpose * (y-X*w)
	d = np.dot(c,b)
	
	#Calculating summation over ((y-X*w)Transpose * (y-X*w))
	err = d.sum()
	N = Xtest.shape[0]
	
	#Calculating error 
	rmse = err/N 
	
	#Calculating Root Mean Squared Error
	rmse = np.sqrt(rmse)
	
	return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  
	
	# error = 1/2 (y - w.T * x)^2 + 1/2 lambd * w.T *w
	# error_grad= (xT.x) * wT - xT.y + lambd * w
    # IMPLEMENT THIS METHOD                           
	# error
	# Reshaping w
	w = w[:,np.newaxis]
	#print w.shape
	
	#Calculating X*w
	Xw = np.dot(X,w)
	#print Xw.shape
	
	#Calculating y-X*w
	z = y - Xw
	#print z.shape
	
	#Calculating (y - Xw)Transpose * (y-X*w)
	term = np.dot(np.transpose(z), z)
	
	jw_1 = term.sum(axis=0)/2
	
	wTw = np.dot(np.transpose(w), w)
	jw_2 = lambd * wTw
	jw_2 = jw_2/2
	
	error = jw_1 + jw_2
	
	# error_grad
	# Calculating (xT.x) * wT
	xT = np.transpose(X)
	xTx = np.dot(xT,X)
	xTx_w = np.dot(xTx, w)
	
	#Calculating xT.y
	xTy = np.dot(xT, y)
	
	#Calculating lambd * w
	lambd_w = np.dot(lambd, w)
	
	error_grad = xTx_w - xTy + lambd_w
	
	#error.flatten()
	#error_grad.flatten()
	
#	return error[:,].flatten(), error_grad[:,].flatten()
	return error.flatten(), error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
	z = p + 1
	q = np.array([x])
	q = np.transpose(q)
	#print q.shape
	N = x.shape[0]
	#print N
	Xd = np.ones_like(q)
	for j in range(1,z):
		res = q ** j
		Xd = np.hstack((Xd,res))
	#print Xd.shape
	return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))     
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()
zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest) 
plt.show()

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

#training_mle = testOLERegression(w,X,y)
#training_mle_intercept = testOLERegression(w_i,X_i,y)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

#print('RMSE without intercept on training data: '+str(training_mle))
#print('RMSE with intercept on training data: '+str(training_mle_intercept))
#plt.plot(w_i)
#plt.show()

# Problem 3
#k = 40
#lambdas = np.linspace(0, 0.09, num=k)
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
#rmses3_train = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
	#rmses3_train[i] = testOLERegression(w_l,X_i,y)
	#print 'Lambda: ' + str(lambd) +  'Test Error: ' + str(rmses3[i]) + 'Train Error: ' + str(rmses3_train[i])
    i = i + 1
#w_l = learnRidgeRegression(X_i,y,0.06)
#mle_i = testOLERegression(w_l,Xtest_i,ytest)
#training_mle_intercept = testOLERegression(w_l,X_i,y)
#w = learnRidgeRegression(X,y,0.06)
#mle = testOLERegression(w,Xtest,ytest)
#training_mle = testOLERegression(w,X,y)
#print('Ridge Regression RMSE with intercept on testing data: '+str(mle_i))
#print('Ridge Regression RMSE with intercept on training data: '+str(training_mle_intercept))
#print('Ridge Regression RMSE without intercept on testing data: '+str(mle))
#print('Ridge Regression RMSE without intercept on training data: '+str(training_mle))
#plt.plot(w_l)
plt.plot(lambdas,rmses3)
plt.show()

# Problem 4
#k = 40
#lambdas = np.linspace(0, 0.09, num=k)
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
#rmses4_training = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
	#rmses4_training[i] = testOLERegression(w_l,X_i,y)
	#print 'Lambda: ' + str(lambd) + '   Test Error: ' + str(rmses4[i]) + '   Train Error: ' + str(rmses4_training[i])
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
	#print 'p = ' + str(p) + ' Zero Lamda Testing Error: ' + str(rmses5[p,0]) + ' Optimal Lambda Testing Error: ' + str(rmses5[p,1]) 
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()