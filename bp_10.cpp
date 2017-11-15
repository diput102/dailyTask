/*
 * bp神经网络程序
 */
#include <iostream>
#include <iomanip>
#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "time.h"
#include <fstream>
#include <string>
using namespace std;

#define N 1934 //学习样本个数
#define IN 1024 //输入层神经元数目
#define HN 100 //隐层神经元数目
#define HC 3 //隐层层数
#define ON 10 //输出层神经元数目
#define Z 200 //旧权值保存-》每次study的权值都保存下来

//全局变量定义
double P[IN]; //单个样本输入数据
double T[ON]; //单个样本教师数据
double U11[HN][IN]; //输入层至第一隐层权值
double U12[HN][HN]; //第一隐层至第二隐层权值
double U23[HN][HN]; //第二隐层至第三隐层权值
double V[ON][HN]; //第三隐层至输出层权值
double X1[HN]; //第一隐层的输入
double X2[HN]; //第二隐层的输入
double X3[HN]; //第三隐层的输入
double Y[ON]; //输出层的输入
double H1[HN]; //第一隐层的输出
double H2[HN]; //第二隐层的输出
double H3[HN]; //第三隐层的输出
double O[ON]; //输出层的输出
double YU_HN1[HN]; //第一隐层的阈值
double YU_HN2[HN]; //第二隐层的阈值
double YU_HN3[HN]; //第三隐层的阈值
double YU_ON[ON]; //输出层的阈值
double err_m[N]; //第m个样本的总误差
double a = 0.1; //学习效率
double alpha = 0.0;  //动量因子
double d_err[ON];//输出层至隐层的一般误差
double e_err3[HN];//第三隐层各神经元的一般误差
double e_err2[HN];//第二隐层各神经元的一般误差
double e_err1[HN];//第一隐层各神经元的一般误差

//结构定义
//学习样本
struct {
	double input[IN]; //输入在上面定义是五个        
	double teach[ON]; //输出在上面定义是三个
}Study_Data[N];

//bp算法权值
struct {
	double old_U11[HN][IN];  //保存输入层至隐层权值旧权
	double old_U12[HN][HN]; //保存第一隐层至第二隐层权值
	double old_U23[HN][HN]; //保存第二隐层至第三隐层权值
	double old_V[ON][HN];  //保存第三隐层至输出层旧权
}Old_WV[Z];

//函数定义
int saveWV(int m)
{
        for(int i=0;i<HN;i++)
        {
                for(int j=0;j<IN;j++)
                {
                        Old_WV[m].old_U11[i][j] = U11[i][j];
                }
        }

        for(int i1=0;i1<HN;i1++)
        {
                for(int j1=0;j1<HN;j1++)
                {
                        Old_WV[m].old_U12[i1][j1] = U12[i1][j1];
                }
        }

        for(int i2=0;i2<HN;i2++)
        {
                for(int j2=0;j2<HN;j2++)
                {
                        Old_WV[m].old_U23[i2][j2] = U23[i2][j2];
                }
        }
        for(int i3=0;i3<ON;i3++)
        {
                for(int j3=0;j3<HN;j3++)
                {
                        Old_WV[m].old_V[i3][j3] = V[i3][j3];
                }
        }
        return 1;
}

int initial()
{//初始化隐层权、阈值
	srand( (unsigned)time( NULL ) );

    for(int i=0;i<HN;i++)
    {//初始化输入层到第一隐层的权值，随机模拟0 和 1 -1
        for(int j=0;j<IN;j++)
		{
             U11[i][j]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int i1=0;i1<HN;i1++)
    {//初始化第一隐层到第二隐层权值
        for(int j1=0;j1<HN;j1++)
		{
             U12[i1][j1]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int i2=0;i2<HN;i2++)
    {//初始化第二隐层到第三隐层权值
        for(int j2=0;j2<HN;j2++)
		{
             U23[i2][j2]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int i3=0;i3<ON;i3++)
    {//初始化隐层到输出层的权值
        for(int j3=0;j3<HN;j3++)
		{
             V[i3][j3]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int k=0;k<HN;k++)
    {//第一隐层阈值初始化 ,-0.01 ~ 0.01 之间
       YU_HN1[k] = (double)((rand()/32767.0)*2-1);  
    }
    for(int k1=0;k1<HN;k1++)
    {//第二隐层阈值初始化
        YU_HN2[k1] = (double)((rand()/32767.0)*2-1);  
    }
    for(int k2=0;k2<HN;k2++)
    {//第三隐层阈值初始化
        YU_HN3[k2] = (double)((rand()/32767.0)*2-1);  
    }
    for(int kk=0;kk<ON;kk++)
    {//输出层阈值初始化
        YU_ON[kk] = (double)((rand()/32767.0)*2-1); 
	}
    return 1;
}

int input_P(int m)
{//输入第m个学习样本
	for (int i=0;i<IN;i++)
	{//获得第m个样本的数据
		P[i]=Study_Data[m].input[i];
	}
	return 1;
}

int input_T(int m)
{//输入第m个输出样本
	for (int k=0;k<ON;k++)
	{
		T[k]=Study_Data[m].teach[k];
	}
	return 1;
}

int H_I_O()
{//输入、输出各隐层值
	double sigma1,sigma2,sigma3;
	int i,i1,i2,j,j1,j2;
	for (j=0;j<HN;j++)
	{
		sigma1=0.0;
		for (i=0;i<IN;i++)
		{//求第一隐层内积
                sigma1+=U11[j][i]*P[i];
		}
		//求第一隐层净输入
		X1[j]=sigma1 - YU_HN1[j];
		//求第一隐层输出sigmoid算法
		H1[j]=1.0/(1.0+exp(-X1[j]));
	}
	//求第二隐层
	for (j1=0;j1<HN;j1++)
	{
		sigma2=0.0;
		for (i1=0;i1<HN;i1++)
		{
                sigma2+=U12[j1][i1]*H1[i1];
		}
		X2[j1]=sigma2 - YU_HN2[j1];
		H2[j1]=1.0/(1.0+exp(-X2[j1]));
	}
	//求第三隐层
	for (j2=0;j2<HN;j2++)
	{
		sigma3=0.0;
		for (i2=0;i2<HN;i2++)
		{
                sigma3+=U23[j2][i2]*H2[i2];
		}
		X3[j2]=sigma3 - YU_HN3[j2];
		H3[j2]=1.0/(1.0+exp(-X3[j2]));
	}
	return 1;
}

int O_I_O()
{//输入、输出各输出层值
	double sigma;
	for (int k=0;k<ON;k++)
	{
		sigma=0.0;
		for (int j=0;j<HN;j++)
		{//求输出层内积
			sigma+=V[k][j]*H3[j];
		}
		//求输出层净输入
		Y[k]=sigma-YU_ON[k];
		//求输出层输出
		O[k]=1.0/(1.0+exp(-Y[k]));
	}
	return 1;
}


int Err_O_H(int m)
{//计算输出层至隐层的一般误差
	double abs_err[ON];//每个样本的绝对误差都是从0开始的
	double sqr_err=0;//每个样本的平方误差计算都是从0开始的
	for (int k=0;k<ON;k++)
	{//求第m个样本下的第k个神经元的绝对误差
		abs_err[k]=T[k]-O[k];
		//求第m个样本下输出层的平方误差
		sqr_err+=(abs_err[k])*(abs_err[k]);
		//d_err[k]输出层各神经元的一般误差
		d_err[k]=abs_err[k]*O[k]*(1.0-O[k]);
	}
	//第m个样本下输出层的平方误差/2=第m个样本的均方误差
	err_m[m]=sqr_err/2;
	return 1;
}

int Err_H_I()
{//计算隐层至输入层的一般误差
	double sigma3,sigma2,sigma1;
	for (int j3=0;j3<HN;j3++) 
	{//第三隐层各神经元的一般误差
		sigma3=0.0;
		for (int k3=0;k3<ON;k3++) 
		{
			sigma3=d_err[k3]*V[k3][j3];
		}
		e_err3[j3]=sigma3*H3[j3]*(1-H3[j3]);
	}
	for (int j2=0;j2<HN;j2++) 
	{//第二隐层各神经元的一般误差
		sigma2=0.0;
		for (int k2=0;k2<HN;k2++) 
		{
			sigma2=d_err[k2]*V[k2][j2];
		}
		e_err2[j2]=sigma2*H2[j2]*(1-H2[j2]);
	}
	for (int j1=0;j1<HN;j1++) 
	{//第一隐层各神经元的一般误差
		sigma1=0.0;
		for (int k1=0;k1<HN;k1++) 
		{
			sigma1=d_err[k1]*V[k1][j1];
		}
		e_err1[j1]=sigma1*H1[j1]*(1-H1[j1]);
	}
	return 1;
}

int Delta_O_H3(int m,int n)
{//调整输出层至第三隐层的权值、输出层阈值
	if(n<=1)
	{
        for (int k=0;k<ON;k++)
        {
            for (int j=0;j<HN;j++)
            {//输出层至第三隐层的权值调整
                 V[k][j]=V[k][j]-a*d_err[k]*H3[j];
            }
			//输出层阈值调整
            YU_ON[k]-=a*d_err[k];
        }
	}
	else if(n>1)
	{
        for (int k=0;k<ON;k++)
        {
            for (int j=0;j<HN;j++)
			{//输出层至隐层的权值调整
				V[k][j] = V[k][j] + a*d_err[k] * H3[j];//+alpha*(V[k][j]-Old_WV[(n-1)].old_V[k][j]);
            }
			//输出层至隐层的阈值调整
            YU_ON[k]-=a*d_err[k];
		 }
	}
	return 1;
}

int Delta_H3_H2(int m,int n)
{//调整第三隐层至第二隐层的权值、第三隐层阈值
	if(n<=1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//第二隐层至第三隐层层的权值调整
                 U23[k][j]=U23[k][j]-a*e_err3[k]*H2[j];
            }
			//第三隐层阈值调整
            YU_HN3[k]-=a*e_err3[k];
        }
	}                
	else if(n>1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//第二隐层至第三隐层层的权值调整
				U23[k][j] = U23[k][j] + a*e_err3[k] * H2[j];// +alpha*(U23[k][j] - Old_WV[(n - 1)].old_U23[k][j]);
            }
			//第三隐层阈值调整
            YU_HN3[k]-=a*e_err3[k];
        }
	}
	return 1;
}

int Delta_H2_H1(int m,int n)
{//调整第二隐层至第一隐层的权值、第二隐层阈值
	if(n<=1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//第一隐层至第二隐层层的权值调整
                 U12[k][j]=U12[k][j]-a*e_err2[k]*H1[j];
            }
			//第二隐层阈值调整
            YU_HN2[k]-=a*e_err2[k];
        }
	}                
	else if(n>1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//第一隐层至第二隐层层的权值调整
				U12[k][j] = U12[k][j] + a*e_err2[k] * H1[j];// +alpha*(U12[k][j] - Old_WV[(n - 1)].old_U12[k][j]);
            }
			//第二隐层阈值调整
            YU_HN2[k]-=a*e_err2[k];
        }
	}
	return 1;
}

int Delta_H1_I(int m,int n)
{//调整第一隐层至输入层的权值、第一隐层阈值
	if(n<=1)
	{
        for (int j=0;j<HN;j++)
        {
            for (int i=0;i<IN;i++) 
            {//第一隐层至输入层的权值调整
                 U11[j][i]=U11[j][i]+a*e_err1[j]*P[i];
            }
			//第一隐层阈值调整
			YU_HN1[j]+=a*e_err1[j];
        }
	}
	else if(n>1)
	{
        for (int j=0;j<HN;j++)
        {
            for (int i=0;i<IN;i++) 
            {//第一隐层至输入层的权值调整
				U11[j][i] = U11[j][i] + a*e_err1[j] * P[i];// +alpha*(U11[j][i] - Old_WV[(n - 1)].old_U11[j][i]);
            }
			//第一隐层阈值调整
			YU_HN1[j]+=a*e_err1[j];
        }
	}
	return 1;
}

double Err_Sum()
{//计算N个样本的全局误差
	double total_err=0;
	for (int m=0;m<N;m++) 
	{//每个样本的均方误差加起来就成了全局误差
		total_err+=err_m[m];
	}
	return total_err;
}

int GetTrainingData()
{//从文件读取输入数据
	ifstream trainTxt("trainingDigits/train.txt");
	std::string fileName;
	for (int n = 0; n < N; n++)
	{
		getline(trainTxt, fileName);
		ifstream data("trainingDigits/" + fileName + ".txt");
		std::cout << fileName + ".txt" << std::endl;
		std::string tempStr;
		for (int i = 0; i < 32; i++)
		{
			getline(data, tempStr);
			for (int j = 0; j < 32; j++)
			{
				Study_Data[n].input[i * 32 + j] = double(tempStr[j] - '0');
			}

		}
		for (int i = 0; i < ON; i++)
		{
			if (i == int(fileName[0] - '0'))
				Study_Data[n].teach[i] = 1.0;
			else
				Study_Data[n].teach[i] = 0.0;
		}

		data.close();
	}
	trainTxt.close();
     //   ifstream GetTrainingData ( "粘度数据.txt", ios::in);
     //   for(int m=0;m<N;m++)
     //   {
     //           for(int i=0;i<IN;i++)
     //           {//取得输入数据
     //               GetTrainingData>>Study_Data[m].input[i];
					//std::cout << Study_Data[m].input[i];
     //           }
     //           for(int j=0;j<ON;j++)
     //           {//取得输出数据
     //               GetTrainingData>>Study_Data[m].teach[j];
     //           }
     //   }
      //  GetTrainingData.close();
        return 1;
}

void cleanfile(string filename)
{//清空文件内容
	ofstream cleanFN(filename, ios::out);
	cleanFN<<"";
	cleanFN.close();
}

void savequan()
{//保存权值
        ofstream outQuanFile("权值.txt", ios::out);
        ofstream outYuFile("阈值.txt", ios::out);
        outQuanFile<<"A\n";
        for(int i=0;i<HN;i++)
        {//输入层至第一隐层权值
                for(int j=0;j<IN;j++)
                {
                        outQuanFile<<U11[i][j]<<"   "; 
                }
                outQuanFile<<"\n";

        }
        outQuanFile<<"\nB\n";
        for(int i1=0;i1<HN;i1++)
        {//第一隐层至第二隐层权值
                for(int j1=0;j1<HN;j1++)
                {
                        outQuanFile<<U12[i1][j1]<<"   ";
                }
                outQuanFile<<"\n";
        }
        outQuanFile<<"\nC\n";
        for(int i2=0;i2<HN;i2++)
        {//第二隐层至第三隐层权值
                for(int j2=0;j2<HN;j2++)
                {
                        outQuanFile<<U23[i2][j2]<<"   ";
                }
                outQuanFile<<"\n";

        }
        outQuanFile<<"\nD\n";
        for(int i3=0;i3<ON;i3++)
        {//第三隐层至输出层权值
                for(int j3=0;j3<HN;j3++)
                {
                        outQuanFile<<V[i3][j3]<<"   ";
                }
                outQuanFile<<"\n";
        }
        outYuFile<<"第一隐层的阈值为:\n";
        for(int k1=0;k1<HN;k1++)
        {//隐层阈值
                outYuFile<<YU_HN1[k1]<<"  ";
        }
        outYuFile<<"\n\n第二隐层的阈值为:\n";
        for(int k2=0;k2<HN;k2++)
        {//隐层阈值
                outYuFile<<YU_HN2[k2]<<"  ";
        }
        outYuFile<<"\n\n第三隐层的阈值为:\n";
        for(int k3=0;k3<HN;k3++)
        {//隐层阈值
                outYuFile<<YU_HN3[k3]<<"  ";
        }
        outYuFile<<"\n\n输出层的阈值为:\n";
        for(int k=0;k<ON;k++)
        {//输出层阈值
                outYuFile<<YU_ON[k]<<"  ";
        }
        outQuanFile.close();
		outYuFile.close();
}

void savewu(int study,double sum_err)
{//保存误差
	ofstream outWuFile( "误差.txt", ios::app);
	//第study次学习的均方误差为sum_err
	outWuFile<<study<<"\t"<<sum_err<<"\n";
	outWuFile.close();
}

//主函数功能
int main()
{
	double sum_err=0;
	int study;//训练次数
	double a = 0.1;//学习速率，即步长0.6
	double alpha = 0.8;  //动量因子
	study=0; //学习次数
	double Pre_error ; //预定误差
	Pre_error = 0.1;
	GetTrainingData();//输入样本
	initial(); //隐层、输出层权、阈值初始化
	cleanfile("权值.txt");
	cleanfile("阈值.txt");
	cleanfile("误差.txt");
	clock_t begin=clock();
	do
	{
		++study;
		if (study % 100 == 0)
			a = a * 0.1;
		for (int m=0;m<N;m++) 
		{//全部样本训练
			input_P(m); //输入第m个学习样本 
			input_T(m);//输入第m个输出样本
			H_I_O(); //第m个学习样本隐层各神经元输入、输出值 
			O_I_O(); //第m个学习样本输出层各神经元输入、输出值
			Err_O_H(m); //第m个学习样本输出层至隐层一般误差  
			Err_H_I(); //第m个学习样本隐层至输入层一般误差 
			Delta_O_H3(m,study); //第m个学习样本输出层至第三隐层权值、阈值调整、修改
			Delta_H3_H2(m,study); //第m个学习样本第三隐层至第二隐层的权值、阈值调整、修改
			Delta_H2_H1(m,study); //第m个学习样本第二隐层至第一隐层的权值、阈值调整、修改
			Delta_H1_I(m,study); //第m个学习样本第一隐层至输入层的权值、阈值调整、修改
		} 
		sum_err=Err_Sum(); //全部样本全局误差计算  
		//saveWV(study);  //把本次的学习权值全保存到数组
		//savewu(study,sum_err);
		std::cout<< "第" + to_string(long long(study)) + "次迭代误差为： " << sum_err / N << std::endl;
	} while (sum_err > Pre_error && study<300); //结束条件误差满足限制或循环最大限
	clock_t end=clock();
	cout<<"样本学习完毕！"<<endl;
	cout<<"bp神经网络已经学习了"<<study<<"次,学习的误差为"<<sum_err<<endl;
	cout <<"总体运行时间为："<<(double)(end - begin)/CLOCKS_PER_SEC<<"秒"<<endl;
	savequan();
	system ("pause");
	return 0;
}