#include  <iostream>
#include  <fstream>
#include  <string>


using   namespace  std;

char maxNum(int a, int b, int c){
	int max=a;
	char Lab='a';
	if(b>max){
		max=b;Lab='g';}
	if(c>max){
		max=c;Lab='s';}
	return Lab;
}


char outputVote(char type[]){
//	//a(asphalt):1,g(grass):2,s(sand):3
	int a=0, g=0, s=0;
	int a1=0,g1=0,s1=0;

	FILE *fp;
	if((fp=fopen("output.txt","r"))==NULL)  //读取txt的文件
	{
		printf("读取文件失败 \n ");
		exit(1);
	}
	//cout<<"读取"<<"成功"<<endl;
	while(!feof(fp))
	{
		switch (fgetc(fp))
		{
		case '1':
			a++;
			break;
		case '2':
			g++;
			break;
		case '3':
			s++;
			break;
		}		
	}
	/*printf("a=%d\n",a);
	printf("g=%d\n",g);
	printf("s=%d\n",s);*/
	fclose(fp);
	//cout<<maxNum(a,g,s);
	for (int j=9;j>0;j--)
	{
		type[j]=type[j-1];		
	}
	type[0]=maxNum(a,g,s);

	for(int j=0;j<10;j++){
		switch (type[j])
		{
		case 'a':
			a1++;
			break;
		case 'g':
			g1++;
			break;
		case 's':
			s1++;
			break;
		default:
			break;
		}
	}

	return maxNum(a1,g1,s1);
}
