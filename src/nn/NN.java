package nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class NN {	
	Matrix size;
	int n;
	public double learningRate;				//learning rate
	int weightPenaltyL2;					//L2 regularization
	int nonSparsityPenalty;					//Non sparsity penalty
	double sparsityTarget;					//Sparsity target
	double inputZeroMaskedFraction;			// Used for Denoising AutoEncoders
	int dropoutFraction;					//Dropout level
	int testing;	
	public double scaling_learningRate;
	public int sae;
	public Matrix []b;						//biases
	public Matrix []W;						//weights
	Matrix []p;								//average activations
	public Matrix e;
	public Matrix []a;
	Matrix []dW;
	Matrix []db;
	
	double L;
	
	
	public int normalize_input;
	public String activation_function;
	
	//nnsetup creates a Feedforward Backpropagate Neural Network
	//����Ϊ���������ز������ľ���
	//���ظ�������
	public NN nnsetup(Matrix Size){
		size=Size.clone();
		n=(int)size.getColumnCount();
		learningRate=0.1;
		weightPenaltyL2=0;
		nonSparsityPenalty=0;
		sparsityTarget=0.05;
		inputZeroMaskedFraction=0;
		dropoutFraction=0;
		testing=0;
		scaling_learningRate=1;
		sae=0;
				
		b=new Matrix[n];
		W=new Matrix[n];
		p=new Matrix[n];	
		
		//��ʼ��������Ԫ��ƫ��ֵ��Ȩ�ء�ƽ�������
		for (int i=1;i<n;++i)
		{
			b[i-1]=DenseMatrix.Factory.zeros(size.getAsLong(0,i),1);
			W[i-1]=DenseMatrix.Factory.rand(size.getAsLong(0,i),size.getAsLong(0,i-1)).minus(0.5).times(2*4*Math.sqrt(6.0/(size.getAsLong(0,i)+size.getAsLong(0,i-1))));
			p[i]=DenseMatrix.Factory.zeros(1,size.getAsLong(0,i));
		}	
		
		return this;
	}
	
	//nntrain trains a neural net
	//ѵ������������x�����y��opts����batchsize��numepochs
	//����ѵ���õ�������
	public NN nntrain(Matrix x, Matrix y, Opt opts){
		opts.silent=0;
		
		long m=x.getRowCount();	
		long col=x.getColumnCount();
		
		int batchsize=opts.batchsize;	//���������������С
		int numepochs=opts.numepochs;	//����ͬһ�����ĵ�������
		
		double []errors=new double[numepochs];	//ÿ�ε������������������ķ������
		
		int numbatches=(int)Math.floor(m/batchsize);	//������������������������С������ȡ�����õ�������
		
		double []LL=new double[numepochs*numbatches];	//ÿ������������ķ������
		
		int N=0;
		
		//��ÿ�ε���
		for (int i=0;i<numepochs;++i)
		{
			//����һ��������Ϊ1��m��������У�mΪ��������
			int []kk=new int[(int)m];
			ArrayList<Integer> list=new ArrayList<Integer>(0);
			for (int l=0;l<m;++l) list.add(l);
			Collections.shuffle(list);
			for (int l=0;l<m;++l) kk[l]=list.get(l);
					
			learningRate=learningRate*scaling_learningRate;
			
			//��ÿһ��������,�������ݿ�
			for (int l=0;l<numbatches;++l)
			{
				Matrix batch_x=DenseMatrix.Factory.zeros(batchsize,col);
				Matrix batch_y=DenseMatrix.Factory.zeros(batchsize,y.getColumnCount());
				if (l==numbatches-1)	//�������һ��,�������batchsize�������ϳ�һ��������
				{
					list.clear();
					for (int j=(numbatches-1)*batchsize;j<m;++j) list.add(kk[j]);
					batch_x=x.selectRows(Ret.NEW, list);
					batch_y=y.selectRows(Ret.NEW, list);
				}
				else					//���ѡȡbatchsize��������ֵ��batch_x��batch_y
				{
					list.clear();
					for (int j=l*batchsize;j<(l+1)*batchsize;++j) list.add(kk[j]);						
					batch_x=x.selectRows(Ret.NEW, list);
					batch_y=y.selectRows(Ret.NEW, list);
				}
				
				//add noise to input
				if (inputZeroMaskedFraction!=0)
				{
					Matrix Rand=DenseMatrix.Factory.rand(batchsize,col);
					for (int t1=0;t1<batchsize;++t1)
						for (int t2=0;t2<col;++t2)
							if (Rand.getAsDouble(t1,t2)<=inputZeroMaskedFraction) batch_x.setAsDouble(0, t1, t2);
				}
				
				//�ݶ��½������ƫ����������W��b
				this.nnff(batch_x,batch_y);		//ǰ������������������ļ���ֵ�����
				this.nnbp();					//�����ݶ��½��㷨���W��b��ƫ����
				this.nnapplygrads();			//����nnbp�����ƫ��������W��b
				
				if (sae==1)
				{
					Matrix tempW=DenseMatrix.Factory.emptyMatrix();
					tempW=W[0].plus(W[1].transpose()).divide(2);
					for (int t1=0;t1<tempW.getColumnCount();++t1)
					{
						W[0]=tempW.clone();
						W[1]=tempW.transpose().clone();
					}
				}
				//�ô�����������ķ������
				LL[N]=L;
				N=N+1;
			}
			
			
			if (opts.silent!=1)
			{ 
				double sum=0;
				for (int t1=N-numbatches;t1<N-1;++t1) sum+=LL[t1]; 
				errors[i]=sum/numbatches;
			}
		}
		
		return this;
	}
	
	//nnff performs a feedforward pass
	public NN nnff(Matrix x,Matrix y){
		long m=x.getRowCount();
		
		a=new Matrix[(int)n];
		a[0]=x;
		
		for (int i=1;i<n;++i)
		{
			//sigm�Ĳ���Ϊͬʱ����m��������W*x+b
			Matrix Extend=DenseMatrix.Factory.repmat(Ret.NEW, b[i-1].transpose(), m, 1);	//repmat(nn.b{i - 1}', m, 1)
			Matrix ones=DenseMatrix.Factory.ones(m,size.getAsLong(0,i));
			Matrix zeros=DenseMatrix.Factory.zeros(m,size.getAsLong(0,i));
			Extend=Extend.plus(a[i-1].mtimes(W[i-1].transpose()));				//repmat(nn.b{i - 1}', m, 1) + nn.a{i - 1} * nn.W{i - 1}'
			Extend=zeros.minus(Extend);			
			a[i]=ones.divide(ones.plus(Extend.exp(Ret.NEW)));			//sigma����
			
			if (dropoutFraction>0 && i<n-1)
			{
				if (testing>0)
					a[i]=a[i].times(1-dropoutFraction);
				else 
				{
					Matrix Rand=DenseMatrix.Factory.rand(m,size.getAsLong(0,i));
					for (int t1=0;t1<m;++t1)
						for (int t2=0;t2<size.getAsLong(0,i);++t2)
							if (Rand.getAsDouble(t1,t2)<=dropoutFraction) 
								a[i].setAsDouble(0, t1, t2);
				}	
			}
			
			if (nonSparsityPenalty>0)
				p[i]=p[i].times(0.99).plus(a[i].mean(Ret.NEW, 1, true).times(0.01));	//��i��ÿһ����Ԫ�ڶ�������µ�ƽ�������
		}
		
		e=y.minus(a[n-1]);		//������
		L=0.5*e.power(Ret.NEW, 2).getValueSum()/m;		//һ������������ķ�����ۺ���
		
		return this;
	}
	
	//nnbp performs backpropagation
	public NN nnbp(){
		dW=new Matrix[n];
		db=new Matrix[n];
		
		Matrix sparsityError;
		
		//d�����ʾ�������Ԫ�Ĳв�,d{n}��ʾ�����n��ÿ����Ԫ�Ĳв�
		Matrix []d=new Matrix[n];
		d[n-1]=e.times(a[n-1].times(a[n-1].minus(1)));
		
		for (int i=n-2;i>0;--i)
		{
			sparsityError=DenseMatrix.Factory.zeros(a[i].getRowCount(),size.getAsLong(0,i));
			Matrix ones=DenseMatrix.Factory.ones(a[i].getRowCount(),a[i].getColumnCount());
			if (nonSparsityPenalty>0)	//��Ҫ����ϡ��������
			{
				Matrix pi=DenseMatrix.Factory.repmat(Ret.NEW, p[i], a[i].getRowCount(), 1);
				sparsityError=ones.times(sparsityTarget).minus(1).divide(pi.minus(1)).minus(ones.times(sparsityTarget).divide(pi)).times(nonSparsityPenalty);
			}
			d[i]=(d[i+1].mtimes(W[i]).plus(sparsityError)).times(a[i].times(ones.minus(a[i])));
		}
		
		//��������ƫ����
		for (int i=0;i<n-1;++i)
		{
			dW[i]=d[i+1].transpose().mtimes(a[i]).divide(d[i+1].getRowCount());
			db[i]=d[i+1].sum(Ret.NEW, 0, true).transpose().divide(d[i+1].getRowCount());
		}
		
		return this;
	}

	//nnapplygrads updates weights and biases with calculated gradients
	public NN nnapplygrads(){
		//�ӵ�һ�㵽�����ڶ���
		for (int i=0;i<n-1;++i)
		{
			W[i]=W[i].minus(dW[i].plus(W[i].times(weightPenaltyL2)).times(learningRate));
			b[i]=b[i].minus(db[i].times(learningRate));
		}
		
		return this;
	}
	
	public void nntest(Matrix x,Matrix y){
		testing=1;
		nnff(x,y);
		testing=0;
		
	}
	
	public void nnchecknumgrad(Matrix x,Matrix y){
		double epsilon=1e-4;
		double er=1e-9;
		double tempdW,tempdb,tempe;
		for (int l=0;l<n-1;++l)
		{
			for (int i=0;i<W[l].getRowCount();++i)
				for (int j=0;j<W[l].getColumnCount();++j)
				{
					NN nn_m=this;
					NN nn_p=this;
					
					nn_m.W[l].setAsDouble(W[l].getAsDouble(i,j)-epsilon,i,j);
					nn_p.W[l].setAsDouble(W[l].getAsDouble(i,j)+epsilon, i,j);
					
					nn_m=nn_m.nnff(x, y);
					nn_p=nn_p.nnff(x, y);
					
					tempdW=(nn_p.L-nn_m.L)/(2*epsilon);
					tempe=Math.abs(tempdW-dW[l].getAsDouble(i,j));
					//if (tempe>er)
				}
			for (int i=0;i<b[l].getRowCount();++i)
			{
				NN nn_m=this;
				NN nn_p=this;
				
				nn_m.b[l].setAsDouble(b[l].getAsDouble(i,0)-epsilon,i,0);
				nn_p.b[l].setAsDouble(b[l].getAsDouble(i,0)+epsilon,i,0);
				
				nn_m.nnff(x, y);
				nn_p.nnff(x, y);
				
				tempdb=(nn_p.L-nn_m.L)/(2*epsilon);
				tempe=Math.abs(tempdb-db[l].getAsDouble(i,0));
				
			}
		}
	}
}
