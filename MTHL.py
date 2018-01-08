# @CIKM: Multi-View Time-Series Hypersphere Learning - MTHL
# @Xian Teng
# @Last Updated 2017-10
"""
-----------------------------------------------

Functions include:

_generate_laplacian_matrix:	generate Laplacian matrix, denoted "Lc" in Eq(8)
_kernel:					calculate kernel of two matrix
_tau:						calculate the weightings tau used in reconstruction error
_GradientDescent:			update P,Q using gradient descent through Eq(16-17)
_mapping:					project X into latent subspece via YY=P.T*X*Q
_compute_R:					compute hypersphere radius (in fact it is radius^2)
_compute_centroid:			compute hypersphere centroid
Optimization:				main route of the alternating optimization algorithm
get_radius:					return radius (radius^2)
predict:					use the trained model to predict new data (label or score)
-----------------------------------------------
"""
import numpy as np
import math
import SVDD

class MTHL(object):

	def __init__(self,X,config):
		
		"""Args:
			X: a list of input tensors with shape [#samples,#features,#timesteps]
			p: reduced feature dimension
			q: reduced time dimension
			lambda1/.lambda2: tradeoff parameters
			gamma: update step
			s: time window
			convergence: iteration convergence tolerance
		"""

		self._X = X
		self._p = config.p
		self._q = config.q
		self._lambda1 = config.lambda1
		self._lambda2 = config.lambda2
		self._gamma = config.gamma
		self._s = config.s
		self._convergence = config.convergence # convergence tolerance (set to 1e-8)
		self._tolerance = config.tolerance # only if d >= r(1+tolerance) we can judge anomalous

		self._V = len(self._X) # number of views
		self._m = len(self._X[0]) # number of samples
		self._T = self._X[0].shape[2] # timesteps
		self._P = [] # a list of matrices for multiple views
		self._Q = []
		self._L = [] # view-specific feature numbers
		for i in range(self._V):
			self._L.append(self._X[i].shape[1]) # number of features for different views
			P = np.random.rand(self._X[i].shape[1],self._p)
			self._P.append(np.linalg.qr(P)[0])
			Q = np.random.rand(self._T,self._q)
			self._Q.append(np.linalg.qr(Q)[0])

		self._Y = np.random.rand(self._p,self._q) # centroid
		self._R = float("inf") # radius
		self._Lp = None # Laplacian matrix

	def _generate_laplacian_matrix(self):
		"""generate Laplacian matrix in Eq(8)
		"""
		g = lambda i,j: 1 if np.abs(i-j)<=np.floor(self._s*0.5) else 0
		weight_matrix = np.array([[g(i,j) for j in range(self._T)] for i in range(self._T)])
		diag_matrix = np.diag(np.sum(weight_matrix, axis=0))
		laplacian_matrix = diag_matrix - weight_matrix
		self._Lp = laplacian_matrix

	# phi(x,y)
	def _kernel(self,x,y):
		return np.trace(np.dot(x,np.transpose(y)))

	# Kernel matrix - KM
	def _gram_matrix(self,YY):
		KM = np.array([[self._kernel(YY[i],YY[j]) for i in range(len(YY))] for j in range(len(YY))])
		return KM

	# calculate tau for one specific view
	def _tau(self,X):
		"""Weighted reconstruction error Eq(10)
		"""
		D = np.array([[math.sqrt(self._kernel(X[i]-X[j],X[i]-X[j])) 
					for j in range(len(X))] for i in range(len(X))])
		Dmean = np.mean(D,axis=0) # mean(D[i,:])
		norm_Dmean = (Dmean - np.mean(Dmean,axis=0)) / np.std(Dmean,axis=0)
		tau = np.array([math.exp(-x) for x in norm_Dmean])
		tau = tau/np.sum(tau)
		return tau

	# calculate tau values for all views
	def _tau2(self):
		return np.array([self._tau(self._X[v]) for v in range(self._V)])

	def _GradientDescent(self,tau):
		"""update P,Q by gradient descent Eq(16)(17)
		"""
		P = [np.zeros((self._L[i], self._p)) for i in range(self._V)]
		Q = [np.zeros((self._T, self._q)) for i in range(self._V)]

		for v in range(self._V):
			for i in range(self._m):
				# update P[v]
				tmp = self._Q[v].T.dot(self._X[v][i,:,:].T).dot(self._P[v]) - self._Y.T
				tmp = self._X[v][i,:,:].dot(self._Q[v]).dot(tmp) * tau[v,i]
				tmp = tmp + self._lambda2*self._X[v][i,:,:].dot(self._Lp).dot(self._X[v][i,:,:].T).dot(self._P[v])
				tmp = tmp * 2
				P[v] = P[v] + tmp
				
				# update Q[v]
				tmp = self._P[v].T.dot(self._X[v][i,:,:]).dot(self._Q[v]) - self._Y
				tmp = self._X[v][i,:,:].T.dot(self._P[v]).dot(tmp)
				tmp = tmp * 2 * tau[v,i]
				Q[v] = Q[v] + tmp

			# update
			P[v] = self._P[v] - self._gamma * P[v]
			Q[v] = self._Q[v] - self._gamma * Q[v]

			# normalize P,Q
			P[v] = np.linalg.qr(P[v])[0]
			Q[v] = np.linalg.qr(Q[v])[0]

		return P,Q

	# bilinear projection via P.T*X*Q => YY
	def _mapping(self,P,Q):
		YY = np.array([[P[v].T.dot(self._X[v][i]).dot(Q[v]) for i in range(self._m)] for v in range(self._V)])
		YY = YY.reshape((self._V*self._m,self._p,self._q)) # [V,m,p,q]
		return YY # length=m*V


	# alpha's length is m*V
	# tau's length is m*V
	def compute_obj(self,alpha,YY,P,Q,tau):
		KM = self._gram_matrix(YY)
		n = len(alpha) # m*V
		# svdd term Eq(23)
		svdd_obj = np.sum(np.array([(tau[i]+alpha[i])*KM[i,i]-np.sum(np.array([(tau[i]+alpha[i])*(tau[j]+alpha[j])*KM[i,j]*0.5 for j in range(n)])) for i in range(n)]))
		# temporal smoothing term Eq(8)
		temp = np.array([[np.trace(P[v].T.dot(self._X[v][i]).dot(self._Lp).dot(self._X[v][i].T).dot(self._P[v])) for i in range(self._m)] for v in range(self._V)])
		obj = svdd_obj + np.sum(temp) * self._lambda2
		
		return obj,KM

	def _compute_centroid(self,YY,alpha,tau):
		"""Eq(21)
		"""
		Y = np.array([YY[i] * (tau[i]+alpha[i]) for i in range(len(YY))])
		Y = np.sum(Y,axis=0) * 0.5
		return Y

	def _compute_R2(self,YY,alpha,tau,KM):
		"""pick up support vectors using Eq(27)
		"""
		diff = [YY[t]-self._Y for t in range(len(YY))]
		r2 = np.array([self._kernel(d,d) for d in diff])
		ids = [t for t in range(len(YY)) if alpha[t] > 0 and alpha[t] < self._lambda1]
		return np.mean(r2[ids])


	def Optimization(self,max_cnt):
		"""Table 2: MTHL algorithm
		"""
		obj_min = float("inf")
		self._generate_laplacian_matrix()
		tau = self._tau2() # [[],[],...]
		trainer = SVDD.SVDDTrainer(SVDD.Kernel.trace(), self._lambda1) # step 2 SVDD

		cnt = 0
		while cnt <= max_cnt:
			P,Q = self._GradientDescent(tau)
			YY = self._mapping(P,Q) # mapping X into YY
			tau1s = tau.reshape((tau.shape[0]*tau.shape[1],1)) # reshape it
			alpha = trainer.train(YY,tau1s)
			obj,KM = self.compute_obj(alpha,YY,P,Q,tau1s)

			if (obj - obj_min) < -self._convergence:  # if smaller, update all
				obj_min = obj
				self._P = P
				self._Q = Q
				cnt += 1
			else:
				# calculate centroid & radius
				Y = self._compute_centroid(YY,alpha,tau1s)
				self._Y = Y
				R2 = self._compute_R2(YY,alpha,tau1s,KM)
				self._R2 = R2
				return

	def get_radius(self):
		return self._R2

	def predict(self,X,fg="score"):
		"""Given a new dataset X, calculate distances (fg="score") or labels (fg="label")
		"""
		diff = [[self._P[v].T.dot(X[v][i]).dot(self._Q[v]) - self._Y for i in range(len(X[0)] for v in range(self._V)]
		dist = np.array([[self._kernel(d,d) for d in diff[v]] for v in range(self._V)])
		
		if fg=="score":
			return dist
		elif fg=="label":
			label = (dist <= self._R2*(1+self._tolerance))
			return np.prod(label,axis=0)
		elif fg=="both":
			label = (dist <= self._R2*(1+self._tolerance))
			return dist,np.prod(label,axis=0)
		else:
			return None



