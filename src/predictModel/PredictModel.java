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
		//读入数据
		for (int i=0;i<35;++i)
		{
			String name="D://data/data"+(i+1)+".csv";
			air_bj[i]=readData(name).clone();
		}
		stationLocation=readData("D://data/stationLocation.csv").clone();
		
		//参数取值数组
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
		
		//每个站邻近的站
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
		

		/*outputStation 循环*/
		int outputStations=1;	//输出站点
		
		
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
		
		/*timeDelay 循环*/
		for (int iTimeDelay=0;iTimeDelay<numTimeDelay.length;++iTimeDelay)
		{
			int timeDelay=numTimeDelay[iTimeDelay];		//要预测几个小时之后的值，0为1个小时。 如果预测步长增长，输入输出和网络参数需要做相应调整
			int range=timeDelay+1;		//输入为之前几个小时的值，如预测8小时：在10点预测18点，输入为3~10小时，输出为18时.如预测1小时：在10点预测11点，输入为10小时，输出为11时
			
			int []inputPolluents={0,1,2,4};	//输入指标
			int []outputPolluents={5};	//输出污染物，5为pm2.5
			
			int []weatherFactors={6,7,8,9,10,11,12,13};
			
			int inputSize=(inputPolluents.length+adj)*range+weatherFactors.length*3*adj;
			int outputSize=1;
			
			int trainingLength=2400;
			int testingLength=240;
			
			Matrix train_x=DenseMatrix.Factory.emptyMatrix();
			Matrix train_y=DenseMatrix.Factory.emptyMatrix();
			
			
			//共有1~trainingLength个训练样本，对每个训练样本i：
			//输入数据为：[i : i+range-1]（也即[i : i+timeDelay]，共range个数据）的污染物数据（自身和adj）
			//加上[i+range : i+range+timeDelay]，共range个天气数据的3个统计值
			//输出数据为：(i+range+timeDelay)的PM2.5值
			//也即在第i个训练样本中，在i+timeDelay时刻，以[i : i+timeDelay]的污染数据和
			//[i+range : i+range+timeDelay]的天气数据预测(i+range+timeDelay)时刻的PM2.5数据
			
			//以i=1，预测8小时后（timeDelay=7）为例：
			//训练样本1中，在第1+7=8小时预测第1+7+8=16小时的PM2.5。
			//输入为[0:7]的污染物数据和[8:15]的天气数据，输出为16小时时的PM2.5数据
			
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
				
				//对每一项天气指标，在一段时间内取三个统计值
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
			
			//测试样本
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
			
			/*确定hiddenNodes层每层的节点数*/
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
					
					/*numEpochs循环*/
					for (int iEpochs=0;iEpochs<numEpochs.length;++iEpochs)
					{
						/*numBatchSize循环-*/
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
								fnn.W[iFnn]=sae.ae[iFnn].W[0];	//赋值为sae中学习到的权重
								fnn.b[iFnn]=sae.ae[iFnn].b[0];	//赋值为sae中学习到的偏置
							}
							
							fnn.nnff(train_x, DenseMatrix.Factory.zeros(train_x.getRowCount(),hiddenStruct.getAsInt(0,hiddenStruct.getColumnCount()-1)));
							Matrix train_feature=fnn.a[fnn.a.length-1];		//展开网络得到特征
							
							
							//普通神经网络，预测模型			
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
							nn.nntrain(train_feature, train_y, opts);	//将特征作为输入
							
							/*将两个网络结合*/
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
							
							
							//保存权重
							Matrix []weights=dnn.W;
							Matrix []bias=dnn.b;
							
							dnn.nnff(test_x, test_y);
							
							Matrix t1=test_y;					//真实数据
							Matrix t2=dnn.a[dnn.a.length-1];	//预测结果
							
							
							
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
