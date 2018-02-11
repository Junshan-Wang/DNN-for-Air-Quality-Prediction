package nn;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class SAE extends NN{
	public NN []ae;
	
	//saesetup creates a  Stacked Auto-Encoders Neural Network
	public SAE saesetup(Matrix Size){
		size=Size.clone();
		n=(int)size.getColumnCount();
		
		ae=new NN[n-1];
		Matrix tmpSize=DenseMatrix.Factory.zeros(1,3);
		for (int u=1;u<n;++u)
		{
			long size0=size.getAsLong(0,u-1);
			long size1=size.getAsLong(0,u);
			tmpSize.setAsLong(size0, 0,0);
			tmpSize.setAsLong(size1, 0,1);
			tmpSize.setAsLong(size0, 0,2);
			
			ae[u-1]=new NN();
			ae[u-1].nnsetup(tmpSize);
		}
		
		return this;
	}
	
	//saetrain trains the Neural Network
	public SAE saetrain(Matrix x, Opt opts){
		//�ֲ�ѵ��
		for (int i=0;i<n-1;++i)
		{
			ae[i].nntrain(x,x,opts);
			NN t=new NN();
			t=ae[i].nnff(x, x);		//������㼤��ֵ��ǰ����㣩
			x=t.a[1];				//���ڶ���ļ���ֵ��Ϊ��һ���Ա�����������
		}
		return this;
	}
	
}
