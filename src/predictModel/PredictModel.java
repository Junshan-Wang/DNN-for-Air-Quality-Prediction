package predictModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import com.csvreader.CsvReader;

public class PredictModel {

	public static void main(String[] args) throws NumberFormatException, IOException {
		Matrix []air_bj=new Matrix[35];
		Matrix stationLocation;
		//��������
		for (int i=0;i<35;++i)
		{
			String name="D://data/data"+(i+1)+".csv";
			air_bj[i]=readData(name).clone();
		}
		stationLocation=readData("D://data/stationLocation.csv").clone();
		
		//����ȡֵ����
		int []numTimeDelay={0};
		int []numHiddenLayer={2};
		int []numHiddenNodes={20};
		int []numEpochs={50};
		int []numBatchSize={10};
		//int []numTimeDelay={47,45,43,41,39,37,35,33,31};
		//int []numHiddenLayer={1,2,3};
		//int []numHiddenNodes={20,50,80,100};
		//int []numEpochs={50,80,100};
		//int []numBatchSize={10,20};
		
		//ÿ��վ�ڽ���վ
		int adj=5;
		Matrix adjMatrix=DenseMatrix.Factory.zeros(35,adj);
		for (int i=0;i<35;++i)
		{		
			List<Double> list1=new ArrayList();
			List<Double> list2=new ArrayList();
			for (int j=0;j<15;++j) 
			{
				double dis=stationLocation.selectRows(Ret.NEW, i).euklideanDistanceTo(stationLocation.selectRows(Ret.NEW, j), true);
				list1.add(dis);
				list2.add(dis);
			}
			list2.sort(null);
			
			for (int j=0;j<adj;++j) adjMatrix.setAsInt(list1.indexOf(list2.get(j)), i,j);
		}
		
		Matrix avgError=DenseMatrix.Factory.zeros(100,1);
		

		/*outputStation ѭ��*/
		int outputStations=1;	//���վ��
		
		
		class MyResult{
			public int timeDelay;
			public int hiddenLayers;
			public Matrix hiddenStruct;
			public int numepochs;
			public int batchsize;
			public int time;
			public double avgError;
			public double mape;
			public Matrix []weights;
			public Matrix []bias;
			public Matrix t1;
			public Matrix t2;

		}
		MyResult []myResult=new MyResult[1];
		int globalCount=0;
		
		/*timeDelay ѭ��*/
		for (int iTimeDelay=0;iTimeDelay<numTimeDelay.length;++iTimeDelay)
		{
			int timeDelay=numTimeDelay[iTimeDelay];		//ҪԤ�⼸��Сʱ֮���ֵ��0Ϊ1��Сʱ�� ���Ԥ�ⲽ��������������������������Ҫ����Ӧ����
			int range=timeDelay+1;		//����Ϊ֮ǰ����Сʱ��ֵ����Ԥ��8Сʱ����10��Ԥ��18�㣬����Ϊ3~10Сʱ�����Ϊ18ʱ.��Ԥ��1Сʱ����10��Ԥ��11�㣬����Ϊ10Сʱ�����Ϊ11ʱ
			
			int []inputPolluents={0,1,2,4};	//����ָ��
			int []outputPolluents={5};	//�����Ⱦ�5Ϊpm2.5
			
			int []weatherFactors={6,7,8,9,10,11,12,13};
			
			int inputSize=(inputPolluents.length+adj)*range+weatherFactors.length*3*adj;
			int outputSize=1;
			
			int trainingLength=2400;
			int testingLength=240;
			
			Matrix train_x=DenseMatrix.Factory.emptyMatrix();
			Matrix train_y=DenseMatrix.Factory.emptyMatrix();
			
			
			//����1~trainingLength��ѵ����������ÿ��ѵ������i��
			//��������Ϊ��[i : i+range-1]��Ҳ��[i : i+timeDelay]����range�����ݣ�����Ⱦ�����ݣ������adj��
			//����[i+range : i+range+timeDelay]����range���������ݵ�3��ͳ��ֵ
			//�������Ϊ��(i+range+timeDelay)��PM2.5ֵ
			//Ҳ���ڵ�i��ѵ�������У���i+timeDelayʱ�̣���[i : i+timeDelay]����Ⱦ���ݺ�
			//[i+range : i+range+timeDelay]����������Ԥ��(i+range+timeDelay)ʱ�̵�PM2.5����
			
			//��i=1��Ԥ��8Сʱ��timeDelay=7��Ϊ����
			//ѵ������1�У��ڵ�1+7=8СʱԤ���1+7+8=16Сʱ��PM2.5��
			//����Ϊ[0:7]����Ⱦ�����ݺ�[8:15]���������ݣ����Ϊ16Сʱʱ��PM2.5����
			
			for(int i=0;i<trainingLength;++i)
			{
				Matrix t=DenseMatrix.Factory.emptyMatrix();
				
				for (int j=0;j<inputPolluents.length;++j)
				{
					t=DenseMatrix.Factory.concat(Ret.NEW, 1, t, air_bj[outputStations].subMatrix(Ret.NEW, i,inputPolluents[j],i+range-1,inputPolluents[j]).transpose());
				}
				for (int j=0;j<adj;++j)
				{
					t=DenseMatrix.Factory.concat(Ret.NEW, 1, t, air_bj[adjMatrix.getAsInt(outputStations,j)].subMatrix(Ret.NEW, i,5,i+range-1,5).transpose());
				}
				
				//��ÿһ������ָ�꣬��һ��ʱ����ȡ����ͳ��ֵ
				for (int j=0;j<weatherFactors.length;++j)
				{
					for (int k=0;k<adj;++k)
					{
						Matrix line=air_bj[adjMatrix.getAsInt(outputStations,k)].subMatrix(Ret.NEW, i+range,weatherFactors[j],i+range+timeDelay,weatherFactors[j]).transpose();
						t=DenseMatrix.Factory.concat(Ret.NEW, 1, t, line.mean(Ret.NEW, 1, true),line.max(Ret.NEW, 1),line.selectColumns(Ret.NEW, timeDelay));
					}
				}
				train_x=DenseMatrix.Factory.concat(Ret.NEW, 0, train_x, t);
				
				t=DenseMatrix.Factory.emptyMatrix();
				for (int j=0;j<1;++j)
				{
					for (int k=0;k<outputPolluents.length;++k)
					{
						t=DenseMatrix.Factory.concat(Ret.NEW, 1, t,air_bj[outputStations].getAsMatrix(i+range+timeDelay,outputPolluents[k]));
					}
				}
				train_y=DenseMatrix.Factory.concat(Ret.NEW, 0, train_y, t);
			}
			
			//��������
			Matrix test_x=DenseMatrix.Factory.emptyMatrix();
			Matrix test_y=DenseMatrix.Factory.emptyMatrix();
			
			for(int i=trainingLength;i<trainingLength+testingLength;++i)
			{
				Matrix t=DenseMatrix.Factory.emptyMatrix();
				
				for (int j=0;j<inputPolluents.length;++j)
				{
					t=DenseMatrix.Factory.concat(Ret.NEW, 1, t, air_bj[outputStations].subMatrix(Ret.NEW, i,inputPolluents[j],i+range-1,inputPolluents[j]).transpose());
				}
				for (int j=0;j<adj;++j)
				{
					t=DenseMatrix.Factory.concat(Ret.NEW, 1, t, air_bj[adjMatrix.getAsInt(outputStations,j)].subMatrix(Ret.NEW, i,5,i+range-1,5).transpose());
				}
				
				for (int j=0;j<weatherFactors.length;++j)
				{
					for (int k=0;k<adj;++k)
					{
						Matrix line=air_bj[adjMatrix.getAsInt(outputStations,k)].subMatrix(Ret.NEW, i+range,weatherFactors[j],i+range+timeDelay,weatherFactors[j]).transpose();
						t=DenseMatrix.Factory.concat(Ret.NEW, 1, t, line.mean(Ret.NEW, 1, true),line.max(Ret.NEW, 1),line.selectColumns(Ret.NEW, timeDelay));
					}
				}
				test_x=DenseMatrix.Factory.concat(Ret.NEW, 0, test_x, t);
				
				t=DenseMatrix.Factory.emptyMatrix();
				for (int j=0;j<1;++j)
				{
					for (int k=0;k<outputPolluents.length;++k)
					{
						t=DenseMatrix.Factory.concat(Ret.NEW, 1, t,air_bj[outputStations].getAsMatrix(i+range+timeDelay,outputPolluents[k]));
					}
				}
				test_y=DenseMatrix.Factory.concat(Ret.NEW, 0, test_y, t);
			}
			
			////*prediction model*////
			
			int hiddenLayers=2;
			
			/*ȷ��hiddenNodes��ÿ��Ľڵ���*/
			for (int index1=0;index1<numHiddenNodes.length;++index1)
			{
				for (int index2=0;index2<numHiddenNodes.length;++index2)
				{
					int hiddenNodes1=50;//numHiddenNodes[index1];
					int hiddenNodes2=20;//numHiddenNodes[index2];
					Matrix hiddenStruct=DenseMatrix.Factory.ones(1,2);
					hiddenStruct.setAsInt(hiddenNodes1, 0,0);
					hiddenStruct.setAsInt(hiddenNodes2, 0,1);
					
					/*sae*/
					nn.SAE sae=new nn.SAE();
					Matrix size=DenseMatrix.Factory.emptyMatrix();
					size=DenseMatrix.Factory.concat(Ret.NEW, 1, DenseMatrix.Factory.ones(1,1).times(inputSize),hiddenStruct);
					sae.saesetup(size);
					for (int iSae=0;iSae<hiddenStruct.getColumnCount();++iSae)
					{
						sae.ae[iSae].normalize_input=0;
						sae.ae[iSae].activation_function="sigm";
						sae.ae[iSae].learningRate=1;
						sae.ae[iSae].scaling_learningRate=0.99;
						sae.ae[iSae].sae=1;
					}
					
					/*numEpochsѭ��*/
					for (int iEpochs=0;iEpochs<numEpochs.length;++iEpochs)
					{
						/*numBatchSizeѭ��-*/
						for (int iBatchSize=0;iBatchSize<numBatchSize.length;++iBatchSize)
						{
							//int startTime=clock;
							nn.Opt opts=new nn.Opt();
							opts.numepochs=numEpochs[iEpochs];
							opts.batchsize=numBatchSize[iBatchSize];
							opts.silent=1;
							sae.saetrain(train_x, opts);
							
							
							nn.NN fnn=new nn.NN();
							size=DenseMatrix.Factory.emptyMatrix();
							size=DenseMatrix.Factory.concat(Ret.NEW, 1,DenseMatrix.Factory.ones(1,1).times(inputSize),hiddenStruct);
							fnn.nnsetup(size);
							
							for (int iFnn=0;iFnn<hiddenStruct.getColumnCount();++iFnn)
							{
								fnn.W[iFnn]=sae.ae[iFnn].W[0];	//��ֵΪsae��ѧϰ����Ȩ��
								fnn.b[iFnn]=sae.ae[iFnn].b[0];	//��ֵΪsae��ѧϰ����ƫ��
							}
							
							fnn.nnff(train_x, DenseMatrix.Factory.zeros(train_x.getRowCount(),hiddenStruct.getAsInt(0,hiddenStruct.getColumnCount()-1)));
							Matrix train_feature=fnn.a[fnn.a.length-1];		//չ������õ�����
							
							
							//��ͨ�����磬Ԥ��ģ��			
							System.out.println("regression");
							nn.NN nn=new nn.NN();
							size=DenseMatrix.Factory.ones(1,2);
							size.setAsInt(hiddenStruct.getAsInt(0,hiddenStruct.getColumnCount()-1), 0,0);
							size.setAsInt(outputSize, 0,1);
							nn.nnsetup(size);
							opts.sae=0;
							opts.numepochs=numEpochs[iEpochs];
							opts.batchsize=numBatchSize[iBatchSize];
							opts.silent=1;
							nn.learningRate=1;
							nn.scaling_learningRate=0.99;
							nn.nntrain(train_feature, train_y, opts);	//��������Ϊ����
							
							/*������������*/
							nn.NN dnn=new nn.NN();
							size=DenseMatrix.Factory.emptyMatrix();
							size=DenseMatrix.Factory.concat(Ret.NEW, 1, DenseMatrix.Factory.ones(1,1).times(inputSize),hiddenStruct,DenseMatrix.Factory.ones(1,1).times(outputSize));
							dnn.nnsetup(size);
							for (int iiSae=0;iiSae<hiddenStruct.getColumnCount();++iiSae)
							{
								dnn.W[iiSae]=sae.ae[iiSae].W[0];
								dnn.b[iiSae]=sae.ae[iiSae].b[0];
							}
							dnn.W[(int)hiddenStruct.getColumnCount()]=nn.W[0];
							dnn.b[(int)hiddenStruct.getColumnCount()]=nn.b[0];
							
							opts.sae=0;
							opts.numepochs=numEpochs[iEpochs];
							opts.batchsize=numBatchSize[iBatchSize];
							opts.silent=1;
							dnn.learningRate=1;
							dnn.scaling_learningRate=0.99;
							dnn.nntrain(train_x, train_y, opts);
							
							
							//����Ȩ��
							Matrix []weights=dnn.W;
							Matrix []bias=dnn.b;
							
							dnn.nnff(test_x, test_y);
							
							Matrix t1=test_y;					//��ʵ����
							Matrix t2=dnn.a[dnn.a.length-1];	//Ԥ����
							
							
							
							avgError.setAsDouble(t2.minus(t1).getAbsoluteValueSum()*500/testingLength, 0,0);
							double mape=t2.minus(t1).getAbsoluteValueSum()/t1.getValueSum();
							
							Matrix tt=DenseMatrix.Factory.concat(Ret.NEW, 1, t1,t2).times(500);
							System.out.println(tt);
							System.out.println(avgError.getAsDouble(0,0));
							
							
						
							/*myResult[globalCount].timeDelay=timeDelay;
							myResult[globalCount].hiddenLayers=hiddenLayers;
							myResult[globalCount].hiddenStruct=hiddenStruct.clone();
							myResult[globalCount].numepochs=opts.numepochs;
							myResult[globalCount].batchsize=opts.batchsize;
							myResult[globalCount].avgError=avgError.getAsDouble(0,0);
							myResult[globalCount].mape=mape;
							myResult[globalCount].weights=weights.clone();
							myResult[globalCount].bias=bias.clone();
							myResult[globalCount].t1=t1.clone();
							myResult[globalCount].t2=t2.clone();*/
							
							globalCount++;
						}
					}
				}
			}
			
		}
		
		
	}
	static Matrix readData(String name) throws NumberFormatException, IOException{
		File inFile=new File(name);
		String inString="";
		double [][]tmp=new double[5000][20];
		int row=0,col=0;
		try{
			BufferedReader reader=new BufferedReader(new FileReader(inFile));
			CsvReader creader=new CsvReader(reader);
			while(creader.readRecord())
			{
				inString=creader.getRawRecord();
				int i=0,last=0;
				int length=inString.length();
				while(i<=length)
				{
					if (i==length)
					{
						tmp[row++][col++]=Double.parseDouble(inString.substring(last,i));
						i++;
						col=0;
					}
					else if (inString.charAt(i)==',') 
					{
						tmp[row][col++]=Double.parseDouble(inString.substring(last, i));
						i++;
						last=i;		
					}
					else i++;
				}
			}
			creader.close();
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}
		
		Matrix data=DenseMatrix.Factory.importFromArray(tmp);
		
		return data;
	}
}
