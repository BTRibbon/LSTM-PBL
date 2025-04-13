#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>
double loss_funtion(double predict[], double pratice[],int n){
    double res=0;
    for(int i=0;i<n;i++){

        res+= pow(predict[i]-pratice[i],2);         // hàm tính loss function bằng cách bình phương các chênh lệch
                                                    // giữa giá trị thực tế và tính toán
    }
    return res;
}
int main(){
    double du_doan[]={1.0,2.0,3.0};
    double thuc_te[]={3.0,2.0,-5.0};
    int n=3;
    double kq=loss_funtion(du_doan,thuc_te,n);
    printf("%.3lf",kq);
}