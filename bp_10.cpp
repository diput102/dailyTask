/*
 * bp���������
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

#define N 1934 //ѧϰ��������
#define IN 1024 //�������Ԫ��Ŀ
#define HN 100 //������Ԫ��Ŀ
#define HC 3 //�������
#define ON 10 //�������Ԫ��Ŀ
#define Z 200 //��Ȩֵ����-��ÿ��study��Ȩֵ����������

//ȫ�ֱ�������
double P[IN]; //����������������
double T[ON]; //����������ʦ����
double U11[HN][IN]; //���������һ����Ȩֵ
double U12[HN][HN]; //��һ�������ڶ�����Ȩֵ
double U23[HN][HN]; //�ڶ���������������Ȩֵ
double V[ON][HN]; //���������������Ȩֵ
double X1[HN]; //��һ���������
double X2[HN]; //�ڶ����������
double X3[HN]; //�������������
double Y[ON]; //����������
double H1[HN]; //��һ��������
double H2[HN]; //�ڶ���������
double H3[HN]; //������������
double O[ON]; //���������
double YU_HN1[HN]; //��һ�������ֵ
double YU_HN2[HN]; //�ڶ��������ֵ
double YU_HN3[HN]; //�����������ֵ
double YU_ON[ON]; //��������ֵ
double err_m[N]; //��m�������������
double a = 0.1; //ѧϰЧ��
double alpha = 0.0;  //��������
double d_err[ON];//������������һ�����
double e_err3[HN];//�����������Ԫ��һ�����
double e_err2[HN];//�ڶ��������Ԫ��һ�����
double e_err1[HN];//��һ�������Ԫ��һ�����

//�ṹ����
//ѧϰ����
struct {
	double input[IN]; //���������涨�������        
	double teach[ON]; //��������涨��������
}Study_Data[N];

//bp�㷨Ȩֵ
struct {
	double old_U11[HN][IN];  //���������������Ȩֵ��Ȩ
	double old_U12[HN][HN]; //�����һ�������ڶ�����Ȩֵ
	double old_U23[HN][HN]; //����ڶ���������������Ȩֵ
	double old_V[ON][HN];  //�������������������Ȩ
}Old_WV[Z];

//��������
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
{//��ʼ������Ȩ����ֵ
	srand( (unsigned)time( NULL ) );

    for(int i=0;i<HN;i++)
    {//��ʼ������㵽��һ�����Ȩֵ�����ģ��0 �� 1 -1
        for(int j=0;j<IN;j++)
		{
             U11[i][j]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int i1=0;i1<HN;i1++)
    {//��ʼ����һ���㵽�ڶ�����Ȩֵ
        for(int j1=0;j1<HN;j1++)
		{
             U12[i1][j1]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int i2=0;i2<HN;i2++)
    {//��ʼ���ڶ����㵽��������Ȩֵ
        for(int j2=0;j2<HN;j2++)
		{
             U23[i2][j2]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int i3=0;i3<ON;i3++)
    {//��ʼ�����㵽������Ȩֵ
        for(int j3=0;j3<HN;j3++)
		{
             V[i3][j3]= (double)((rand()/32767.0)*2-1);
		}
    }
    for(int k=0;k<HN;k++)
    {//��һ������ֵ��ʼ�� ,-0.01 ~ 0.01 ֮��
       YU_HN1[k] = (double)((rand()/32767.0)*2-1);  
    }
    for(int k1=0;k1<HN;k1++)
    {//�ڶ�������ֵ��ʼ��
        YU_HN2[k1] = (double)((rand()/32767.0)*2-1);  
    }
    for(int k2=0;k2<HN;k2++)
    {//����������ֵ��ʼ��
        YU_HN3[k2] = (double)((rand()/32767.0)*2-1);  
    }
    for(int kk=0;kk<ON;kk++)
    {//�������ֵ��ʼ��
        YU_ON[kk] = (double)((rand()/32767.0)*2-1); 
	}
    return 1;
}

int input_P(int m)
{//�����m��ѧϰ����
	for (int i=0;i<IN;i++)
	{//��õ�m������������
		P[i]=Study_Data[m].input[i];
	}
	return 1;
}

int input_T(int m)
{//�����m���������
	for (int k=0;k<ON;k++)
	{
		T[k]=Study_Data[m].teach[k];
	}
	return 1;
}

int H_I_O()
{//���롢���������ֵ
	double sigma1,sigma2,sigma3;
	int i,i1,i2,j,j1,j2;
	for (j=0;j<HN;j++)
	{
		sigma1=0.0;
		for (i=0;i<IN;i++)
		{//���һ�����ڻ�
                sigma1+=U11[j][i]*P[i];
		}
		//���һ���㾻����
		X1[j]=sigma1 - YU_HN1[j];
		//���һ�������sigmoid�㷨
		H1[j]=1.0/(1.0+exp(-X1[j]));
	}
	//��ڶ�����
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
	//���������
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
{//���롢����������ֵ
	double sigma;
	for (int k=0;k<ON;k++)
	{
		sigma=0.0;
		for (int j=0;j<HN;j++)
		{//��������ڻ�
			sigma+=V[k][j]*H3[j];
		}
		//������㾻����
		Y[k]=sigma-YU_ON[k];
		//����������
		O[k]=1.0/(1.0+exp(-Y[k]));
	}
	return 1;
}


int Err_O_H(int m)
{//����������������һ�����
	double abs_err[ON];//ÿ�������ľ������Ǵ�0��ʼ��
	double sqr_err=0;//ÿ��������ƽ�������㶼�Ǵ�0��ʼ��
	for (int k=0;k<ON;k++)
	{//���m�������µĵ�k����Ԫ�ľ������
		abs_err[k]=T[k]-O[k];
		//���m��������������ƽ�����
		sqr_err+=(abs_err[k])*(abs_err[k]);
		//d_err[k]��������Ԫ��һ�����
		d_err[k]=abs_err[k]*O[k]*(1.0-O[k]);
	}
	//��m��������������ƽ�����/2=��m�������ľ������
	err_m[m]=sqr_err/2;
	return 1;
}

int Err_H_I()
{//����������������һ�����
	double sigma3,sigma2,sigma1;
	for (int j3=0;j3<HN;j3++) 
	{//�����������Ԫ��һ�����
		sigma3=0.0;
		for (int k3=0;k3<ON;k3++) 
		{
			sigma3=d_err[k3]*V[k3][j3];
		}
		e_err3[j3]=sigma3*H3[j3]*(1-H3[j3]);
	}
	for (int j2=0;j2<HN;j2++) 
	{//�ڶ��������Ԫ��һ�����
		sigma2=0.0;
		for (int k2=0;k2<HN;k2++) 
		{
			sigma2=d_err[k2]*V[k2][j2];
		}
		e_err2[j2]=sigma2*H2[j2]*(1-H2[j2]);
	}
	for (int j1=0;j1<HN;j1++) 
	{//��һ�������Ԫ��һ�����
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
{//��������������������Ȩֵ���������ֵ
	if(n<=1)
	{
        for (int k=0;k<ON;k++)
        {
            for (int j=0;j<HN;j++)
            {//����������������Ȩֵ����
                 V[k][j]=V[k][j]-a*d_err[k]*H3[j];
            }
			//�������ֵ����
            YU_ON[k]-=a*d_err[k];
        }
	}
	else if(n>1)
	{
        for (int k=0;k<ON;k++)
        {
            for (int j=0;j<HN;j++)
			{//������������Ȩֵ����
				V[k][j] = V[k][j] + a*d_err[k] * H3[j];//+alpha*(V[k][j]-Old_WV[(n-1)].old_V[k][j]);
            }
			//��������������ֵ����
            YU_ON[k]-=a*d_err[k];
		 }
	}
	return 1;
}

int Delta_H3_H2(int m,int n)
{//���������������ڶ������Ȩֵ������������ֵ
	if(n<=1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//�ڶ�����������������Ȩֵ����
                 U23[k][j]=U23[k][j]-a*e_err3[k]*H2[j];
            }
			//����������ֵ����
            YU_HN3[k]-=a*e_err3[k];
        }
	}                
	else if(n>1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//�ڶ�����������������Ȩֵ����
				U23[k][j] = U23[k][j] + a*e_err3[k] * H2[j];// +alpha*(U23[k][j] - Old_WV[(n - 1)].old_U23[k][j]);
            }
			//����������ֵ����
            YU_HN3[k]-=a*e_err3[k];
        }
	}
	return 1;
}

int Delta_H2_H1(int m,int n)
{//�����ڶ���������һ�����Ȩֵ���ڶ�������ֵ
	if(n<=1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//��һ�������ڶ�������Ȩֵ����
                 U12[k][j]=U12[k][j]-a*e_err2[k]*H1[j];
            }
			//�ڶ�������ֵ����
            YU_HN2[k]-=a*e_err2[k];
        }
	}                
	else if(n>1)
	{
        for (int k=0;k<HN;k++)
        {
            for (int j=0;j<HN;j++)
            {//��һ�������ڶ�������Ȩֵ����
				U12[k][j] = U12[k][j] + a*e_err2[k] * H1[j];// +alpha*(U12[k][j] - Old_WV[(n - 1)].old_U12[k][j]);
            }
			//�ڶ�������ֵ����
            YU_HN2[k]-=a*e_err2[k];
        }
	}
	return 1;
}

int Delta_H1_I(int m,int n)
{//������һ������������Ȩֵ����һ������ֵ
	if(n<=1)
	{
        for (int j=0;j<HN;j++)
        {
            for (int i=0;i<IN;i++) 
            {//��һ������������Ȩֵ����
                 U11[j][i]=U11[j][i]+a*e_err1[j]*P[i];
            }
			//��һ������ֵ����
			YU_HN1[j]+=a*e_err1[j];
        }
	}
	else if(n>1)
	{
        for (int j=0;j<HN;j++)
        {
            for (int i=0;i<IN;i++) 
            {//��һ������������Ȩֵ����
				U11[j][i] = U11[j][i] + a*e_err1[j] * P[i];// +alpha*(U11[j][i] - Old_WV[(n - 1)].old_U11[j][i]);
            }
			//��һ������ֵ����
			YU_HN1[j]+=a*e_err1[j];
        }
	}
	return 1;
}

double Err_Sum()
{//����N��������ȫ�����
	double total_err=0;
	for (int m=0;m<N;m++) 
	{//ÿ�������ľ������������ͳ���ȫ�����
		total_err+=err_m[m];
	}
	return total_err;
}

int GetTrainingData()
{//���ļ���ȡ��������
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
     //   ifstream GetTrainingData ( "ճ������.txt", ios::in);
     //   for(int m=0;m<N;m++)
     //   {
     //           for(int i=0;i<IN;i++)
     //           {//ȡ����������
     //               GetTrainingData>>Study_Data[m].input[i];
					//std::cout << Study_Data[m].input[i];
     //           }
     //           for(int j=0;j<ON;j++)
     //           {//ȡ���������
     //               GetTrainingData>>Study_Data[m].teach[j];
     //           }
     //   }
      //  GetTrainingData.close();
        return 1;
}

void cleanfile(string filename)
{//����ļ�����
	ofstream cleanFN(filename, ios::out);
	cleanFN<<"";
	cleanFN.close();
}

void savequan()
{//����Ȩֵ
        ofstream outQuanFile("Ȩֵ.txt", ios::out);
        ofstream outYuFile("��ֵ.txt", ios::out);
        outQuanFile<<"A\n";
        for(int i=0;i<HN;i++)
        {//���������һ����Ȩֵ
                for(int j=0;j<IN;j++)
                {
                        outQuanFile<<U11[i][j]<<"   "; 
                }
                outQuanFile<<"\n";

        }
        outQuanFile<<"\nB\n";
        for(int i1=0;i1<HN;i1++)
        {//��һ�������ڶ�����Ȩֵ
                for(int j1=0;j1<HN;j1++)
                {
                        outQuanFile<<U12[i1][j1]<<"   ";
                }
                outQuanFile<<"\n";
        }
        outQuanFile<<"\nC\n";
        for(int i2=0;i2<HN;i2++)
        {//�ڶ���������������Ȩֵ
                for(int j2=0;j2<HN;j2++)
                {
                        outQuanFile<<U23[i2][j2]<<"   ";
                }
                outQuanFile<<"\n";

        }
        outQuanFile<<"\nD\n";
        for(int i3=0;i3<ON;i3++)
        {//���������������Ȩֵ
                for(int j3=0;j3<HN;j3++)
                {
                        outQuanFile<<V[i3][j3]<<"   ";
                }
                outQuanFile<<"\n";
        }
        outYuFile<<"��һ�������ֵΪ:\n";
        for(int k1=0;k1<HN;k1++)
        {//������ֵ
                outYuFile<<YU_HN1[k1]<<"  ";
        }
        outYuFile<<"\n\n�ڶ��������ֵΪ:\n";
        for(int k2=0;k2<HN;k2++)
        {//������ֵ
                outYuFile<<YU_HN2[k2]<<"  ";
        }
        outYuFile<<"\n\n�����������ֵΪ:\n";
        for(int k3=0;k3<HN;k3++)
        {//������ֵ
                outYuFile<<YU_HN3[k3]<<"  ";
        }
        outYuFile<<"\n\n��������ֵΪ:\n";
        for(int k=0;k<ON;k++)
        {//�������ֵ
                outYuFile<<YU_ON[k]<<"  ";
        }
        outQuanFile.close();
		outYuFile.close();
}

void savewu(int study,double sum_err)
{//�������
	ofstream outWuFile( "���.txt", ios::app);
	//��study��ѧϰ�ľ������Ϊsum_err
	outWuFile<<study<<"\t"<<sum_err<<"\n";
	outWuFile.close();
}

//����������
int main()
{
	double sum_err=0;
	int study;//ѵ������
	double a = 0.1;//ѧϰ���ʣ�������0.6
	double alpha = 0.8;  //��������
	study=0; //ѧϰ����
	double Pre_error ; //Ԥ�����
	Pre_error = 0.1;
	GetTrainingData();//��������
	initial(); //���㡢�����Ȩ����ֵ��ʼ��
	cleanfile("Ȩֵ.txt");
	cleanfile("��ֵ.txt");
	cleanfile("���.txt");
	clock_t begin=clock();
	do
	{
		++study;
		if (study % 100 == 0)
			a = a * 0.1;
		for (int m=0;m<N;m++) 
		{//ȫ������ѵ��
			input_P(m); //�����m��ѧϰ���� 
			input_T(m);//�����m���������
			H_I_O(); //��m��ѧϰ�����������Ԫ���롢���ֵ 
			O_I_O(); //��m��ѧϰ������������Ԫ���롢���ֵ
			Err_O_H(m); //��m��ѧϰ���������������һ�����  
			Err_H_I(); //��m��ѧϰ���������������һ����� 
			Delta_O_H3(m,study); //��m��ѧϰ�������������������Ȩֵ����ֵ�������޸�
			Delta_H3_H2(m,study); //��m��ѧϰ���������������ڶ������Ȩֵ����ֵ�������޸�
			Delta_H2_H1(m,study); //��m��ѧϰ�����ڶ���������һ�����Ȩֵ����ֵ�������޸�
			Delta_H1_I(m,study); //��m��ѧϰ������һ������������Ȩֵ����ֵ�������޸�
		} 
		sum_err=Err_Sum(); //ȫ������ȫ��������  
		//saveWV(study);  //�ѱ��ε�ѧϰȨֵȫ���浽����
		//savewu(study,sum_err);
		std::cout<< "��" + to_string(long long(study)) + "�ε������Ϊ�� " << sum_err / N << std::endl;
	} while (sum_err > Pre_error && study<300); //������������������ƻ�ѭ�������
	clock_t end=clock();
	cout<<"����ѧϰ��ϣ�"<<endl;
	cout<<"bp�������Ѿ�ѧϰ��"<<study<<"��,ѧϰ�����Ϊ"<<sum_err<<endl;
	cout <<"��������ʱ��Ϊ��"<<(double)(end - begin)/CLOCKS_PER_SEC<<"��"<<endl;
	savequan();
	system ("pause");
	return 0;
}