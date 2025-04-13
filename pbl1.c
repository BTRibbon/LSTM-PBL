#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>
// sigmoid function
float sigmoid(float x){
    
    return 1/(1+exp(-1.0*x));

}

// tanh function
float tah(float x){
    return (exp(x)-exp(-1.0*x))/(exp(x)+exp(-1.0*x));
}
// hàm forget gate
float forget_gate(float W_f,float U_f,float b_f,float x_t,float ht_1){
    return sigmoid(W_f*x_t+U_f*ht_1+b_f);
}
// hàm input gate
float input_gate(float W_i,float U_i,float b_i,float x_t,float ht_1){
    return sigmoid(W_i*x_t+U_i*ht_1+b_i);
}
// hàm ouput gate
float ouput_gate(float W_o,float U_o,float b_o,float x_t,float ht_1){
    return sigmoid(W_o*x_t+U_o*ht_1+b_o);
}
// hàm tính candidate cell
float cell_update(float W_c,float U_c,float b_c,float x_t,float ht_1){
    return tah(W_c*x_t+U_c*ht_1+b_c);
}
// hàm tính final cell state( Ct ) bằng cách lấy cổng forget_gate*C_t-1 với cell_update*input_gate
float final_cell(float ct_1,float f_g,float i_g,float c_update){
    return f_g*ct_1+i_g*c_update;
}
// hàm tính H_t bằng cách lấy output_gate*tanh(final_cell)
float new_ht(float out,float ct_1,float f_g,float i_g,float c_update){

    return out*tah(final_cell( ct_1,f_g,i_g,c_update));

}
typedef struct {
    double Wf, Wi, Wc, Wo; // Trọng số 
    double bf, bi, bc, bo; // Độ lệch
    double h_prev, C_prev; // Trạng thái ẩn và trạng thái ô trước đó
}LSTM;

LSTM test(LSTM tao){

    tao.Wf=1;tao.Wi=1;tao.Wc=1;tao.Wo=1;// tạo trọng số ban đầu cho LSTM

    tao.bf=0;tao.bi=0;tao.bc=0;tao.bo=0;// tạo bias ban đầu cho LSTM

    return tao;
    
}

int main(){


}