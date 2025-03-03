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
// ham forget gate
float forget_gate(float W_f,float U_f,float b_f,float x_t,float ht_1){
    return sigmoid(W_f*x_t+U_f*ht_1+b_f);
}
// ham input gate
float input_gate(float W_i,float U_i,float b_i,float x_t,float ht_1){
    return sigmoid(W_i*x_t+U_i*ht_1+b_i);
}
// ham ouput gate
float ouput_gate(float W_o,float U_o,float b_o,float x_t,float ht_1){
    return sigmoid(W_o*x_t+U_o*ht_1+b_o);
}
// ham tinh candidate cell
float cell_update(float W_c,float U_c,float b_c,float x_t,float ht_1){
    return tah(W_c*x_t+U_c*ht_1+b_c);
}
// ham tinh final cell state( Ct ) bang cach lay cong forget_gate*C_t-1 voi cell_update*input_gate
float final_cell(float ct_1,float f_g,float i_g,float c_update){
    return f_g*ct_1+i_g*c_update;
}
// ham tinh H_t bang cach lay output_gate*tanh(final_cell)
float new_ht(float out,float ct_1,float f_g,float i_g,float c_update){

    return out*tah(final_cell( ct_1,f_g,i_g,c_update));

}
int main(){


}