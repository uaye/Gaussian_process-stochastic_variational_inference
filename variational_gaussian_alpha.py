
import numpy as np
from numpy import linalg as LA
import scipy 
from scipy import stats
from scipy import special
import random
import math


class VA_LBK():
	def __init__(self, train_data_x, train_data_y, test_data_x, test_data_y, lr, initial_val, batch_size, latent_sample_size,epoch):
		self.train_data=train_data_x
		self.train_data_y=train_data_y
		self.test_data_x=test_data_x
		self.test_data_y=test_data_y
		self.lr=lr
		self.batch_size=batch_size
		self.latent_sample_size=latent_sample_size
		self.param_initial=initial_val
		self.epoch=epoch
		self.para_store={}
		
	def prepare_train_data(self):
		x=self.train_data
		(N,D)=x.shape
		pre=np.zeros((N,N,D))
		for i in range(N):
			for j in range(i+1):
				for k in range(D):
					if i!=j:
						pre[i,j,k]=pre[j,i,k]=(x[i,k]-x[j,k])**2
					else:
						pre[i,i,k]=(x[i,k])**2
		return(pre)
	
	def preprare_test_data(self, x_test):
		(n1,d)=x_test.shape
		n2=self.train_data.shape[0]
		pred=np.zeros((n1,n2,d))
		for i in range(n1):
			for j in range(n2):
				for k in range(d):
					pred[i,j,k]=(x_test[i,k]-self.train_data[j,k])**2
		return pred

	
	def kernel_LB(self, x, y, z, a, beta, alpha):
		y1=(1+np.dot(x,a))**beta
		y2=(1+np.dot(y,a))**beta
		y3=(1+np.dot(z,a))**beta
		cov=alpha*(y1+y2-y3-1)
		return(cov)
	
	def covariance_build(self,x_pre,a,beta, alpha):
		n=x_pre.shape[0]
		d=x_pre.shape[2]
		z_t=np.zeros(d)
		kernel=np.zeros((n,n))
		for i in range(n):
			for j in range(i+1):
				if i!=j:
					kernel[i,j]=kernel[j,i]=\
					self.kernel_LB(x_pre[i,i,:],x_pre[j,j,:],x_pre[i,j,:],a,beta, alpha)
				else:
					kernel[i,i]=self.kernel_LB(x_pre[i,i,:],x_pre[i,i,:],z_t,a,beta, alpha)		
		return(kernel)
		
	def log_joint_distribution(self, K, a, p, sigma, sigma_0, y):
		tem=K+np.identity(K.shape[0])*(sigma*2+0.01)
		tem_inv=LA.pinv(tem)
		a_0=(a==0)
		a_1=(a>0)
		t_val=-np.log(LA.det(tem))-0.5*np.dot(y, tem_inv).dot(y)#-\
		#0.5*(np.sum(np.square(np.log(a[a_1])))+(np.log(sigma))**2)/sigma_0**2+np.sum(np.log(p[a_1]))
		#print(LA.det(tem),np.log(LA.det(tem)),t_val)
		return(t_val)
	
	def deri_a(self, mu, phi, p, a, t):
		if t==0:
			return([0.0,0.0, -1/np.min((1-p, 1-1e-5))], np.log(1-p))
		else:
			t=np.log(a)-mu
			d_mu=t/phi**2
			d_phi=-1.0/phi+t**2/4.0/phi**3
			d_p=1.0/np.max((p, 1e-5))
			q=-0.5*(np.log(2*np.pi)+2*np.log(phi)+t**2/phi**2)
			return([d_mu, d_phi, d_p], q)
	
	def deri_beta(self, beta, gam_1, gam_2):
		s1=np.log(gam_1+gam_2)-0.5/(gam_1+gam_2)
		d_gam_1=s1-np.log(gam_1)+0.5/gam_1+np.log(beta)
		d_gam_2=s1-np.log(gam_2)+0.5/gam_2+np.log(1-beta)
		c=scipy.special.beta(gam_1,gam_2)
		q=(gam_1-1)*np.log(beta)+(gam_2-1)*np.log(1-beta)-np.log(c)
		return([d_gam_1,d_gam_2],q)
	
	def deri_sigma(self, sigma, mu_s, phi_s):
		t=np.log(sigma)-mu_s
		d_mu_s=t/phi_s**2
		d_phi_s=-1.0/phi_s+t**2/phi_s**3/4
		q=-0.5*(np.log(2*np.pi)+2*np.log(phi_s)+t**2/phi_s**2)
		return([d_mu_s,d_phi_s],q)
	
	def sample_batch(self, X_pre, y_train):
		N=X_pre.shape[0]
		idx=random.sample(range(N),self.batch_size)
		x_sample=X_pre[idx,:][:,idx]
		y_sample=y_train[idx]
		return(x_sample, y_sample)
	
	def sample_latent_var(self, a_para, beta_para, sigma_para, alpha_para):
		D=len(a_para)
		a_sample=np.zeros((self.latent_sample_size,D,2))
		beta_sample=np.zeros(self.latent_sample_size)
		sigma_sample=np.zeros(self.latent_sample_size)
		alpha_sample=np.zeros(self.latent_sample_size)
		for i in range(self.latent_sample_size):
			beta_sample[i]=min(max(1e-2,scipy.stats.beta.rvs(beta_para[0],beta_para[1])),1-1e-2)
			sigma_sample[i]=np.exp(np.random.randn(1)*sigma_para[1]+sigma_para[0])
			alpha_sample[i]=max(1.0,np.exp(np.random.randn(1)*alpha_para[1]+alpha_para[0]))
			for j in range(D):
				a_sample[i,j,1]=(np.random.uniform()<a_para[j][2])
				if a_sample[i,j,1]:
					a_sample[i,j,0]=np.exp(np.random.randn(1)*a_para[j][1]+a_para[j][0])
				else:
					a_sample[i,j,0]=0
		return(a_sample,beta_sample,sigma_sample,alpha_sample)
		
	def deri_calculate(self, deri_para, log_para, log_joint):
		n_a=int(self.latent_sample_size/1.5)
		f=np.multiply(deri_para, (log_joint-log_para))
		tem=np.row_stack((f, deri_para))[:,range(n_a)]
		tem_cov=np.cov(tem)
		a_d=tem_cov[0,1]/max(tem_cov[1,1],1e-5)
		deri=np.average(f-a_d*deri_para)
		return(deri)
		
	def initiate_parameter(self):
		D=self.train_data.shape[1]
		a_para=np.array([self.param_initial['a_init'] for _ in range(D)])
		beta_para=np.array(self.param_initial['beta_init'])
		sigma_para=np.array(self.param_initial['sigma_init'])
		alpha_para=np.array(self.param_initial['alpha_para'])
		return(a_para, beta_para, sigma_para,alpha_para)
		
	def train(self):
		X_pre=self.prepare_train_data()
		d=self.train_data.shape[1]
		[rho, dec_percent, dec_step]=self.lr
		(a_para, beta_para, sigma_para, alpha_para)=self.initiate_parameter()
		sigma_0=sigma_para[1]
		for itr in range(self.epoch):
			if itr%dec_step==0:
				rho*=dec_percent
			#generate fitting points
			(x_pre_sample, y_sample)=self.sample_batch(X_pre, self.train_data_y)
			#print(x_pre_sample)
			(a_sample,beta_sample,sigma_sample,alpha_sample)=self.sample_latent_var(a_para, beta_para, sigma_para,alpha_para)
			#store temporary log_p
			tem_log_joint=np.zeros(self.latent_sample_size)
			a_para_deri=np.zeros((self.latent_sample_size,d,3))
			q_a=np.zeros((self.latent_sample_size,d))
			beta_para_deri=np.zeros((self.latent_sample_size,2))
			q_beta=np.zeros(self.latent_sample_size)
			sigma_para_deri=np.zeros((self.latent_sample_size,2))
			q_sigma=np.zeros(self.latent_sample_size)
			alpha_para_deri=np.zeros((self.latent_sample_size,2))
			q_alpha=np.zeros(self.latent_sample_size)			
			#calculate all needed derivatives for each sample
			for i in range(self.latent_sample_size):
				K=self.covariance_build(x_pre_sample,a_sample[i,:,0],beta_sample[i],alpha_sample[i])
				tem_log_joint[i]=self.log_joint_distribution(K, a_sample[i,:,0], a_para[:,2], sigma_sample[i], sigma_0, y_sample)
				beta_para_deri[i,:],q_beta[i]=self.deri_beta(beta_sample[i], beta_para[0], beta_para[1])
				alpha_para_deri[i,:],q_alpha[i]=self.deri_sigma(alpha_sample[i], alpha_para[0], alpha_para[1])
				sigma_para_deri[i,:],q_sigma[i]=self.deri_sigma(sigma_sample[i], sigma_para[0], sigma_para[1])
				for j in range(d):
					a_para_deri[i,j,:],q_a[i,j]=self.deri_a(a_para[j,0],a_para[j,1], a_para[j,2], a_sample[i,j,0], a_sample[i,j,1])
			#updating each parameters:
			a_para_old=a_para.copy()
			beta_para_old=beta_para.copy()
			alpha_para_old=alpha_para.copy()
			sigma_para_old=sigma_para.copy()
			for j in range(d):
				t_0=rho*self.deri_calculate(a_para_deri[:,j,0], q_a[:,j], tem_log_joint)
				a_para[j,0]+=np.sign(t_0)*min(abs(t_0),0.1)
				t_1=rho*self.deri_calculate(a_para_deri[:,j,1], q_a[:,j], tem_log_joint)
				t_1=np.sign(t_1)*min(abs(t_1),0.1)
				a_para[j,1]=min(max(t_1+a_para[j,1],1e-5),0.25)
				t_2=rho*self.deri_calculate(a_para_deri[:,j,2], q_a[:,j], tem_log_joint)
				t_2=np.sign(t_2)*min(abs(t_2),0.05)
				a_para[j,2]=min(a_para[j,2]+t_2,1-1e-5)
				a_para[j,2]=max(1e-5, a_para[j,2])
			b_t_1=rho*self.deri_calculate(beta_para_deri[:,0], q_beta, tem_log_joint)
			b_t_1=np.sign(b_t_1)*min(abs(b_t_1),0.1)
			b_1=max(beta_para[0]+b_t_1,0.01)
			b_t_2=rho*self.deri_calculate(beta_para_deri[:,1], q_beta, tem_log_joint)
			b_t_2=np.sign(b_t_2)*min(abs(b_t_2),0.1)
			b_2=max(beta_para[1]+b_t_2,0.01)
			beta_para=np.array([b_1,b_2])
			s_t_1=rho*self.deri_calculate(sigma_para_deri[:,0], q_sigma, tem_log_joint)
			s_t_1=np.sign(s_t_1)*min(0.1,abs(s_t_1))
			sigma_para[0]=min(-1.0, sigma_para[0]+s_t_1)
			s_t_2=rho*self.deri_calculate(sigma_para_deri[:,1], q_sigma, tem_log_joint)
			s_t_2=np.sign(s_t_2)*min(abs(s_t_2),0.1)
			sigma_para[1]=min(max(sigma_para[1]+s_t_2,1e-5),0.1)
			
			alpha_t_1=rho*self.deri_calculate(alpha_para_deri[:,0], q_alpha, tem_log_joint)
			alpha_t_1=np.sign(alpha_t_1)*min(0.1,abs(alpha_t_1))
			alpha_para[0]=min(alpha_para[0]+alpha_t_1,1.0)
			alpha_t_2=rho*self.deri_calculate(alpha_para_deri[:,1], q_alpha, tem_log_joint)
			alpha_t_2=np.sign(alpha_t_2)*min(abs(alpha_t_2),0.1)
			alpha_para[1]=min(max(alpha_para[1]+alpha_t_2,1e-5),0.1)
			
			diff=np.sqrt((np.sum(np.square(a_para_old-a_para))+np.sum(np.square(beta_para_old-beta_para))+\
			np.sum(np.square(sigma_para_old-sigma_para))+np.sum(np.square(alpha_para_old-alpha_para)))/(a_para.shape[0]*3+4)/rho)
			print(itr,np.mean(tem_log_joint),diff,rho)
			if diff<0.005:
				break
		
		self.para_store['a_para']=a_para
		self.para_store['beta_para']=beta_para
		self.para_store['sigma_para']=sigma_para
		self.para_store['alpha_para']=alpha_para
		
		
		
	def post_para_estimate(self):
		post_estimation={}
		a=self.para_store['a_para']
		beta=self.para_store['beta_para']
		sigma=self.para_store['sigma_para']
		alpha=self.para_store['alpha_para']
		post_estimation['a']=np.multiply(a[:,2],np.exp(a[:,0]))
		post_estimation['beta']=beta[0]/(beta[0]+beta[1])
		post_estimation['sigma']=np.exp(sigma[0])
		post_estimation['alpha']=np.exp(alpha[0])
		return(post_estimation)
			

	def prediction_r_build(self, test_x, x_new_prepare, x_pre, a, beta, alpha):
		n1=x_new_prepare.shape[0]
		n2=x_pre.shape[0]
		r=np.zeros((n1,n2))
		for i in range(n1):
			for j in range(n2):
				r[i,j]=\
				self.kernel_LB(np.square(test_x[i,:]), x_pre[j,j,:], x_new_prepare[i,j,:], a, beta, alpha)
		return(r)

			
	def prediction(self, test_x):
		x_new_prepare=self.preprare_test_data(x_test)
		x_pre=self.prepare_train_data()
		post_estimation=self.post_para_estimate()
		r=self.prediction_r_build(test_x, x_new_prepare, x_pre, post_estimation['a'], post_estimation['beta'], post_estimation['alpha'])
		covariance=self.covariance_build(x_pre,post_estimation['a'], post_estimation['beta'], post_estimation['alpha'])
		inv=LA.pinv(covariance+np.identity(r.shape[1])*post_estimation['sigma']**2)
		y_pre=np.dot(np.dot(r,inv),self.train_data_y)
		return(y_pre)
		
	def prediction_average(self, test_x):
		a_para=self.para_store['a_para']
		beta_para=self.para_store['beta_para']
		sigma_para=self.para_store['sigma_para']
		alpha_para=self.para_store['alpha_para']
		x_new_prepare=self.preprare_test_data(x_test)
		x_pre=self.prepare_train_data()
		
		#print(x_pre_sample)
		(a_sample,beta_sample,sigma_sample,alpha_sample)=self.sample_latent_var(a_para, beta_para, sigma_para,alpha_para)		
		f_pre=np.zeros((self.latent_sample_size,len(test_x)))
		for i in range(self.latent_sample_size):
			(x_pre_sample, y_sample)=self.sample_batch(x_pre, self.train_data_y)
			K=self.covariance_build(x_pre_sample,a_sample[i,:,0],beta_sample[i],alpha_sample[i])
			r=self.prediction_r_build(test_x, x_new_prepare, x_pre_sample, a_sample[i,:,0], beta_sample[i], alpha_sample[i])
			inv=LA.pinv(K+np.identity(len(K))*sigma_sample[i]**2)
			f_pre[i,:]=np.dot(np.dot(r,inv),y_sample)
		return(np.average(f_pre, axis=0))
	
np.random.seed(120)		
train_data=np.loadtxt(open("x_train.txt"),delimiter=",")
x_0=train_data[0,:]
train_data=(train_data-x_0)[1:,:]
y=np.loadtxt(open("y_train.txt"),delimiter=",")
y_0=y[0]
y=(y[1:]-y_0)
x_test=np.loadtxt(open("x_test.txt"),delimiter=",")-x_0
y_test=np.loadtxt(open("y_test.txt"),delimiter=",")-y_0
init_val={'a_init':[2.0,0.25,0.3],'beta_init':[1,1],'sigma_init':[-2.0,0.1],'alpha_para':[0.0,0.25]}	
test=VA_LBK(train_data_x=train_data, train_data_y=y, test_data_x=None, test_data_y=None, lr=[0.001, 0.95, 100], \
	batch_size=30, latent_sample_size=80, initial_val=init_val, epoch=6000)

#print(test.covariance_build(self,x_pre,a,beta))
test.train()
print(test.para_store)
y_pre=test.prediction(x_test)
print(np.std(y_pre-y_test))
y_pre_2=test.prediction_average(x_test)
print(np.std(y_pre_2-y_test))