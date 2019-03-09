import csv
import random
import math
import operator


def loadDataset(filename, splitter ,header_line="y"):
	trainingSet=[]
        with open(filename, 'rb') as csvfile:
                lines = csv.reader(csvfile, delimiter=splitter)
		line=[]
		v_rowcount=0
                for row in lines:
			if header_line=="y" and v_rowcount==0:
				pass
			else:
				line=[float(x)*1 for x in row]
				trainingSet.append(line)
			v_rowcount+=1
	return trainingSet

def CenterDataset(points,x_dim,y_dim):
	for i in range(0,x_dim):
		mn=0
		mx=0
		for j in range(0,y_dim):
			if j==0:
				mn=points[j][i]
				mx=points[j][i]
			else:
				if points[j][i]>mx:
					mx=points[j][i]
				if points[j][i]<mn:
					mn=points[j][i]
		#print 'i: %(i)d\tj: %(j)d\tmn: %(mn)f\tmx: %(mx)f'%{"i":i, "j":j, "mn":mn, "mx":mx}
		if mn !=0  or mx !=0:
			for j in range(0,y_dim):
				points[j][i]=(points[j][i]-mx)/(mn-mx)
	return points	

def compute_mse_for_points(b, m, points, x_dim, y_dim):
	totalError = 0
	x = []
	for i in range(0, y_dim):
		x_sum = 0
		x[:] = []
		y = points[i][x_dim]
		for j in range(0, x_dim):
			x.append(points[i][j]) # !!! Y-value s supposed to be in the last-right column
		for j in range(0, x_dim):
			x_sum += m[j] * x[j]
		totalError += (y - (x_sum + b)) ** 2
		#print 'i: %(i)d; x_sum: %(x_sum)f; totalError: %(totalError)f' % {"i":i, "x_sum":x_sum, "totalError":totalError}
	return totalError / float(y_dim)

def gradient(b, m, points, x_dim, y_dim):
	new_m=[]
	new_b=0
	for k in range(0,x_dim): #k=[0,1,...] i.e.: columns
		v_total_sum=0
		for j in range(0,y_dim):
			y=points[j][x_dim]
			v_sum=0
			for i in range(0,x_dim):
				v_sum+=((points[j][i]*m[i]+b)-y)
			v_sum=v_sum*points[j][k]
			#print 'k: %(k)i; j: %(j)d; v_sum: %(v_sum)f' % {"k":k, "j":j, "v_sum":v_sum}
			v_total_sum+=v_sum
		new_m.append(v_total_sum*2/y_dim)
	v_total_sum=0
	for j in range(0,y_dim):
		y=points[j][x_dim]
		v_sum=0
		for i in range(0,x_dim):
			v_sum+=((points[j][i]*m[i]+b)-y)
		v_total_sum+=v_sum
	new_b=(v_total_sum*2/y_dim)
	return new_b,new_m



def main():
	points=[]
	points=loadDataset("gd_dataset.csv", ";")
	#print len(points)
	
	x_dim = len(points[0])-1
	y_dim = len(points)
	learning_rate = -0.01
	num_iterations = 1000

	print 'x_dim: %(x_dim)i; y_dim: %(y_dim)i; learning_rate: %(learning_rate)f' % {"x_dim":x_dim, "y_dim":y_dim, "learning_rate":learning_rate}
	points=CenterDataset(points,x_dim,y_dim)

	b = 1	#bias
	m = []	#
	new_b=0
	new_m=[]
	grad_len_limit=0.01
	for i in range(x_dim):
	    m.append(0)

	for j in range(0,num_iterations):
		mse=compute_mse_for_points(b,m,points,x_dim,y_dim)
		new_b,new_m=gradient(b, m, points, x_dim, y_dim)
		grad_len=0
		for i in range(x_dim):
			grad_len+=(new_m[i]**2)

		grad_len+=new_b**2
		grad_len=grad_len**(1/2.0)
	
		if j>0 and grad_len/prev_grad_len < 1:
			learning_rate=learning_rate*0.8
		prev_grad_len=grad_len
				

		print '#: %(j)d\tmse: %(mse)f\tglen: %(grad_len)f\tlrate: %(learning_rate)f'%{"j":j,"mse":mse, "grad_len":grad_len, "learning_rate":learning_rate}

		for i in range(0,x_dim):
			m[i]=m[i]+learning_rate*new_m[i]

		b=b+learning_rate*new_b
		
		if grad_len<=grad_len_limit:
			break

	print b,m
	
		
main()
