#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

using namespace std;

struct Sample {
    vector<double> input;  // struct ví dụ 
    double output;
};

double activateSigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));     // hàm sigmoid 
}

vector<double> activateSigmoid(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = activateSigmoid(vec[i]);           // dùng hàm sigmoid cho từng phần tử trong vectơ
    return res;
}

double activateTanh(double x) {
    return tanh(x);                  // hàm tanh
}

vector<double> activateTanh(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)         // dùng hàm tanh cho từng phần tử trong vectơ 
        res[i] = activateTanh(vec[i]);
    return res;
}

double dActivateSigmoid(double y) {
    return y * (1.0 - y);                   // đạo hàm của sigmoid
}

vector<double> dActivateSigmoid(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = vec[i] * (1.0 - vec[i]);           // đạo hàm sigmoid của từng phần tử trong vector
    return res;
}

double dActivateTanh(double y) {
    return 1.0 - y * y;                 // đạo hàm của hàm tanh
}

vector<double> dActivateTanh(const vector<double>& vec) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = 1.0 - vec[i] * vec[i];         // đạo hàm của hàm tanh cho từng phần tử trong vector
    return res;
}

vector<double> addVectors(const vector<double>& a, const vector<double>& b) {
    vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); i++)
        res[i] = a[i] + b[i];               // hàm cộng hai vector
    return res;
}

vector<double> multiplyVectors(const vector<double>& a, const vector<double>& b) {
    vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); i++)       // hàm nhân hai vector
        res[i] = a[i] * b[i];
    return res;
}

vector<double> scaleVector(const vector<double>& vec, double factor) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)         
        res[i] = vec[i] * factor;               // hàm nhân vector với một số
    return res;
}

vector<double> matrixVectorProduct(const vector<vector<double>>& mat, const vector<double>& vec) {
    vector<double> res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); i++)
        for (size_t j = 0; j < vec.size(); j++)
            res[i] += mat[i][j] * vec[j];           // hàm nhân một ma trận với một vector( hay là ma trận 1*n)
    return res;
}

vector<double> concatVectors(const vector<double>& v1, const vector<double>& v2) {
    vector<double> res = v1;                             // hàm chèn vector v2 vào cuối vector v1

    res.insert(res.end(), v2.begin(), v2.end());        // res.end() ( hay v1.end() ) là vị trí bắt đầu chèn
                                                        // chèn các phần tử từ v2.begin()->v2.end()
    return res;                                     
}

double clipValue(double val, double minVal, double maxVal) {

    return max(minVal, min(val, maxVal));       // hàm này để giúp ta giới hạn hẹp dần hoặc bằng

    // vd : clipValue(10,0,100) ->max(0,min(10,100)) -> 10. Vậy giới han từ 0-100 giờ thành 10-100
}

vector<double> clipVector(const vector<double>& vec, double minVal, double maxVal) {
    vector<double> res(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        res[i] = clipValue(vec[i], minVal, maxVal);     // làm hẹp giới hạn trong vector
    return res;
}

class MyLSTM {
    int inSize;
    int hidSize;
    vector<vector<double>> Wf, Wi, Wc, Wo;  // các trọng số (weight)
    vector<double> bf, bi, bc, bo;          // các độ lệch (bias)
    
    // Lưu trữ các kết quả trung gian
    vector<double> lastInput, lastHidden, lastCell;
    vector<double> gateForget, gateInput, candidate, gateOutput;
    vector<double> cellState, hiddenState;
    
    mt19937 rng;        // đay là engine sinh số ngẫu nhiên
    normal_distribution<double> dist;       // còn cái này là để sinh ra các giá trị nằm trong khoảng nào đó
    
public:
    MyLSTM(int inputSize, int hiddenSize) 
        : inSize(inputSize), hidSize(hiddenSize), rng(random_device{}()), dist(0.0, 0.1) {

        // cái "inSize(inputSize), hidSize(hiddenSize)"  đó thực ra là :
        // inSize=inputSize; hidSize= hiddenSize; lý do nên dùng cái trên là gọn với hiệu năng cao hơn
        // cái "rng(random_device{}()), dist(0.0, 0.1)"" cũng y vậy
        
        Wf = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        // vector<double>(inSize + hidSize) : câu lệnh này có nghĩa là tạo một vector có
        // inSize + hidSize phần tử và tất cả đều có giá trị bằng 0
        //inSize ở đây chính là số lượng feature của đối tượng. Ví dụ: một ngày có lượng nhiệt độ, độ ẩm
        //  => có 2 feature => inSize = 2
        // trong trường hợp này ta sẽ thấy Wf là ma trận [hidSize*(inSize+hidenSize)]
        Wi = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        Wc = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        Wo = vector<vector<double>>(hidSize, vector<double>(inSize + hidSize));
        bf = vector<double>(hidSize, 0.0);
        bi = vector<double>(hidSize, 0.0);
        bc = vector<double>(hidSize, 0.0);
        bo = vector<double>(hidSize, 0.0);
        
        // Khởi tạo trọng số với giá trị nhỏ ngẫu nhiên
        for (int i = 0; i < hidSize; i++) {
            for (int j = 0; j < inSize + hidSize; j++) {
                Wf[i][j] = dist(rng);
                Wi[i][j] = dist(rng);
                Wc[i][j] = dist(rng);
                Wo[i][j] = dist(rng);
            }
            bf[i] = 1.0; // Thiết lập bias của forget gate
        }
    }
    
    // Hàm forward cho một bước thời gian
    // Hay còn gọi là hàm để tính kết quả với những trọng số ngẫu nhiên đã tạo ban đầu
    pair<vector<double>, vector<double>> forward(const vector<double>& input, 
                                                   const vector<double>& prevHidden, 
                                                   const vector<double>& prevCell) {
        lastInput = input;
        lastHidden = prevHidden;
        lastCell = prevCell;
        vector<double> combined = concatVectors(input, prevHidden);// nối cái X_t với h_t-1 theo công thức
        // và ta sẽ có được [x_t,h_t-1] là ma trận [(inSize+hidenSize)*1]

        gateForget = activateSigmoid(addVectors(matrixVectorProduct(Wf, combined), bf));
        gateInput  = activateSigmoid(addVectors(matrixVectorProduct(Wi, combined), bi));
        candidate  = activateTanh(addVectors(matrixVectorProduct(Wc, combined), bc));
        cellState  = addVectors(multiplyVectors(gateForget, prevCell),
                                 multiplyVectors(gateInput, candidate));
        gateOutput = activateSigmoid(addVectors(matrixVectorProduct(Wo, combined), bo));
        hiddenState = multiplyVectors(activateTanh(cellState), gateOutput);
        
        return {hiddenState, cellState};
        // addVectors: hàm cộng hai vector
        // matrixVectorProduct: hàm nhân một ma trận với một vector( hay là ma trận n*1)
        // multiplyVectors: hàm nhân hai vector
    }

    // Hàm backward cho một bước thời gian
    // dHiddenNext: gradient của hidden state từ bước sau
    // dCellNext: gradient của cell state từ bước sau
    // lr: learning rate

    //this is insane tbh
    // backward: back propagation
    tuple<vector<double>, vector<double>, vector<double>> backward(const vector<double>& dHiddenNext, 
                                                                     const vector<double>& dCellNext, 
                                                                     double lr, 
                                                                     double clipVal = 1.0) {
        vector<double> dGateOutput = multiplyVectors(dHiddenNext, activateTanh(cellState));
        // tính (đạo hàm của L theo h_t) * (đạo hàm của h_t theo o_t)
        // hay là  (đạo hàm của L theo h_t) * (tanh(C_t))
        // công thức giống với word 

        vector<double> dCellTotal = addVectors(dCellNext, 
                                    multiplyVectors(dHiddenNext, 
                                        multiplyVectors(gateOutput, dActivateTanh(activateTanh(cellState)))));
        // từ lúc này CellTotal sẽ là C_t
        // Tính tổng đạo hàm theo C_t:
        // ∂L/∂C_t = dCellNext + dHiddenNext * o_t * (1 - tanh(C_t)^2)C_t

        vector<double> dGateInput  = multiplyVectors(dCellTotal, candidate);
        // Tính ∂L/∂i_t = ∂L/∂C_t * C̃_t
        vector<double> dCandidate  = multiplyVectors(dCellTotal, gateInput);
        // Tính ∂L/∂C̃_t = ∂L/∂C_t * i_t
        vector<double> dGateForget = multiplyVectors(dCellTotal, lastCell);
        // Tính ∂L/∂f_t = ∂L/∂C_t * C_{t-1}

        // Áp dụng đạo hàm của các hàm kích hoạt
        dGateOutput = multiplyVectors(dGateOutput, dActivateSigmoid(gateOutput));
        // kết quả của dGateOutput trên kia nhân với (đạo hàm của cổng ouput theo Wo)

        dGateInput  = multiplyVectors(dGateInput, dActivateSigmoid(gateInput));
        dCandidate  = multiplyVectors(dCandidate, dActivateTanh(candidate));
        dGateForget = multiplyVectors(dGateForget, dActivateSigmoid(gateForget));
        
        // Cắt gradient cho an toàn
        dGateOutput = clipVector(dGateOutput, -clipVal, clipVal);
        dGateInput  = clipVector(dGateInput, -clipVal, clipVal);
        dCandidate  = clipVector(dCandidate, -clipVal, clipVal);
        dGateForget = clipVector(dGateForget, -clipVal, clipVal);
        
        vector<double> combined = concatVectors(lastInput, lastHidden);// kết hợp [x_t,h_t-1]
        
        // Cập nhật trọng số cho mỗi cổng
        // theo công thức W= W- ∂L/∂W * [x_t,h_{t-1}]
        // và code dưới này là để thực hiện phép tính đó
        for (int i = 0; i < hidSize; i++) {
            for (int j = 0; j < inSize + hidSize; j++) {
                Wf[i][j] -= lr * dGateForget[i] * combined[j];
                Wi[i][j] -= lr * dGateInput[i]  * combined[j];
                Wc[i][j] -= lr * dCandidate[i]  * combined[j];
                Wo[i][j] -= lr * dGateOutput[i] * combined[j];
            }
            bf[i] -= lr * dGateForget[i];
            bi[i] -= lr * dGateInput[i];
            bc[i] -= lr * dCandidate[i];
            bo[i] -= lr * dGateOutput[i];
        }
        
        // Tính gradient cho hidden state và cell state của bước trước
        vector<double> dCombined(inSize + hidSize, 0.0);
        // tạo một vector dCombined với (inSize + hidSize) = 0.0

        for (int i = 0; i < hidSize; i++) {
            for (int j = 0; j < inSize + hidSize; j++) {
                dCombined[j] += Wf[i][j] * dGateForget[i] +
                                Wi[i][j] * dGateInput[i] +
                                Wc[i][j] * dCandidate[i] +
                                Wo[i][j] * dGateOutput[i];
            }
        }
        vector<double> dPrevInput(dCombined.begin(), dCombined.begin() + inSize);
        vector<double> dPrevHidden(dCombined.begin() + inSize, dCombined.end());
        vector<double> dPrevCell = multiplyVectors(dCellTotal, gateForget);
        
        // trả về gradient của input, hidden state và cell state.
        // Gradient trọng số có thể được tính riêng nếu cần.
        return {dPrevInput, dPrevHidden, dPrevCell};
    }
};

//Lớp LSTM Model
class LSTMModel {
    //Thông số kiến trúc mạng
    int inputDim;
    int hiddenDim;
    int outputDim;
    MyLSTM lstmCell;

    //Tham số lớp đầu ra
    vector<vector<double>> Wy;
    vector<double> by;

    //Tham số huấn luyện
    double learningRate;
    int epochs;
    
public:
    //Hàm khởi tạo
    LSTMModel(int inDim, int hidDim, int outDim, double lr = 0.01, int ep = 50)
        : inputDim(inDim), hiddenDim(hidDim), outputDim(outDim), lstmCell(inDim, hidDim),
          learningRate(lr), epochs(ep) {
        // cái dòng trên là đến gắn các giá trị vào các tham số inputDim,hiddenDim,....
        // Khởi tạo ngẫu nhiên trọng số Wy và bias by
        // lstmCell(inDim, hidDim): câu này là nó sẽ tạo một lstmCell có các trọng số ngẫu nhiên
        
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> d(0.0, sqrt(6.0/(1.0*(inputDim+hiddenDim))));// Phân phối chuẩn
            
        Wy = vector<vector<double>>(outputDim, vector<double>(hiddenDim));
        // ý nghĩa của Wy : vì hiddensate là vector nhiều chiều mà kết quả( output) thường chỉ là 1 hay 2
        // ==> cái Wy là để giúp hidden cho ra đúng chừng nớ output
        // ví dụ : dự đoán thời tiết thì kết quả chỉ có 1 là nắng, mưa, giông,... mà hiddenstate là vector nhiều chiều
        // ==> Wy ở đây sẽ giúp cho hiddestate sẽ cho ra kết quả là nắng, mưa ,giông,...
        // công thức y(output) = hidden * Wy +by
        by = vector<double>(outputDim, 0.0);

        // Gán giá trị ngẫu nhiên cho Wy
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < hiddenDim; j++)
                Wy[i][j] = d(gen);
        }
    }

    //Hàm huấn luyện mô hình
    double trainModel(const vector<Sample>& dataset) {
        double totalLoss = 0.0;
        
        //vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
        // Lặp qua số lần huấn luyện
        for (int ep = 0; ep < epochs; ep++) {
            double epLoss = 0.0;
            
            double current_lr = learningRate ;
            //vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
            // Duyệt qua từng mẫu trong tập dữ liệu
            for (const auto& s : dataset) {
                
                // Khởi tạo hidden state và cell state ban đầu
                vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
                
                // Lan truyền thuận qua LSTM cell
                auto [hOut, cOut] = lstmCell.forward(s.input, h, c);
                //h=hOut;c=cOut;
                // Lớp output
                vector<double> pred(outputDim, 0.0);
                vector<double> temp = matrixVectorProduct(Wy, hOut);// chỗ này là đang nhân hidden với Wy
                for (int i = 0; i < outputDim; i++)
                    pred[i] = temp[i] + by[i];
                
                // Tính loss theo MSE
                double loss = 0.0;
                vector<double> dPred(outputDim, 0.0);
                for (int i = 0; i < outputDim; i++) {
                    double err = pred[i] - s.output;
                    loss += err * err;
                    dPred[i] = 2.0 * err;
                }
                loss /= outputDim;
                epLoss += loss;
                
                // Lan truyền ngược lớp output
                vector<double> dHidden(hiddenDim, 0.0);

                // Cập nhật trọng số Wy và bias by
                for (int i = 0; i < outputDim; i++) {
                    for (int j = 0; j < hiddenDim; j++) {
                        Wy[i][j] -= current_lr * dPred[i] * hOut[j];
                        dHidden[j] += Wy[i][j] * dPred[i];
                    }
                    by[i] -= current_lr * dPred[i];
                }

                //Lan truyền ngược qua LSTM cell
                vector<double> dCell(hiddenDim, 0.0);// tạo một vector dCell có hiddenDim phần tử = 0

                lstmCell.backward(dHidden, dCell, current_lr);
                // dòng này chủ yếu dùng backward để cập nhật trọng số của lstmCell thôi
                
            }
            // Tính loss trung bình epoch
            epLoss /= dataset.size();
            totalLoss = epLoss;
            
            
            // In thông tin sau mỗi 10 epoch
            if ((ep + 1) % 10 == 0 || ep == 0)
                cout << "Epoch " << (ep + 1) << "/" << epochs << " - Loss: " << epLoss << endl;
        }
        return totalLoss;
    }

    // Hàm dự đoán đầu ra từ dữ liệu mới
    double predictOutput(const vector<Sample>& inputData,int i) {
        vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
        
        // Lan truyền thuận qua LSTM
        // Lan truyền thuận qua LSTM
        auto [hOut, cOut] = lstmCell.forward(inputData[i].input, h, c);//câu lệnh này có nghĩa là:
        // vì hàm forward sẽ trả về 2 giá trị hiddenstate và cellstate
        // nên câu lệnh này hOut sẽ lưu giá trị của hiddenstate và cOut sẽ lưu giá trị cảu cellstate
        
        // Tính toán đầu ra
        vector<double> out = matrixVectorProduct(Wy, hOut);
        for (int i = 0; i < outputDim; i++)
            out[i] += by[i];
        return out[0]; // Giả sử outputDim = 1 hay là kết quả chỉ cần in ra một thôi
    }
    double predictOutput_S(struct Sample e) {
        vector<double> h(hiddenDim, 0.0), c(hiddenDim, 0.0);
        
        // Lan truyền thuận qua LSTM
        // Lan truyền thuận qua LSTM
        auto [hOut, cOut] = lstmCell.forward(e.input, h, c);//câu lệnh này có nghĩa là:
        // vì hàm forward sẽ trả về 2 giá trị hiddenstate và cellstate
        // nên câu lệnh này hOut sẽ lưu giá trị của hiddenstate và cOut sẽ lưu giá trị cảu cellstate

        // Tính toán đầu ra
        vector<double> out = matrixVectorProduct(Wy, hOut);
        for (int i = 0; i < outputDim; i++)
            out[i] += by[i];
        return out[0]; // Giả sử outputDim = 1 hay là kết quả chỉ cần in ra một thôi
    }
        
    
};

void normalize(vector<Sample>& data,double lon){
    int n=data.size();
    
    for(int i=0;i<n;i++){
        for(int j=0;j<3;j++){
            data[i].input[j]/=lon;
        }
        data[i].output/=lon;
    }
}
double fibo(int n) {
    if (n <= 1)
        return n;
    return fibo(n - 1) + fibo(n - 2);
}
void logNormalize(vector<Sample>& data) {   // hàm chuẩn hóa dữ kiệu theo log
    for (auto& sample : data) {
        for (auto& val : sample.input) {
            val = log(val + 1);  // +1 để tránh log(0)
        }
        sample.output = log(sample.output + 1);
    }
}
int main() {
    // Tham số mô hình
    int inDim = 3;
    int hidDim = 64;
    int outDim = 1;
    double lr = 0.005;
    int numEpochs = 4000;

    // Đọc dữ liệu từ file
    vector<Sample> trainData;
    vector<Sample> testInput;
    freopen("D:\\PBL1\\fibodata.txt", "r", stdin);
    double temp;struct Sample e;double a, b,c;
    cin>>a;cin>>b;cin>>c;
    e.input.push_back(a);
    e.input.push_back(b);
    e.input.push_back(c);
    e.output=b+c;
    trainData.push_back(e);
    testInput.push_back(e);
    for(int i=3;i<15;i++){
        struct Sample e;
        temp=b;a=b;b=c;
        c+=temp;
        e.input.push_back(a);
        e.input.push_back(b);
        e.input.push_back(c);
        e.output=b+c;
        trainData.push_back(e);
        testInput.push_back(e);
    }
    struct Sample t;
    temp=b;a=b;b=c;
    c+=temp;
    t.input.push_back(a);
    t.input.push_back(b);
    t.input.push_back(c);
    t.output=b+c;
    testInput.push_back(t);
    //int n=trainData.size();double lon=trainData[n-1].output;
    for(int i=0;i<testInput.size();i++){
        cout<<testInput[i].input[0]<<" "<<testInput[i].input[1]<<" "<<testInput[i].input[2]<<" "<<testInput[i].output<<endl;
    }
    logNormalize(testInput);// chuẩn hóa dự liệu theo log vì do số quá lớn
    logNormalize(trainData);
    
    // Khởi tạo và huấn luyện mô hình
    LSTMModel model(inDim, hidDim, outDim, lr, numEpochs);
    double finalLoss = model.trainModel(trainData);
    cout << "Training completed. Final Loss: " << finalLoss << endl;

    // Dự đoán
    cout << "Testing prediction:" << endl;
    
    
    // Đọc dữ liệu test (5 đặc trưng)
    /*double k=5;
    struct Sample p_thu;
    cout<<fibo(k)<<" "<<fibo(k+1)<<" "<<fibo(k+2)<<endl;
    p_thu.input.push_back(fibo(k)/lon);
    p_thu.input.push_back(fibo(k+1)/lon);
    p_thu.input.push_back(fibo(k+2)/lon);*/
    
    // Gọi hàm dự đoán và in kết quả
    for(int i=0;i<testInput.size();i++){
        double pred = model.predictOutput(testInput,i);
        cout<<"predict(da chuan hoa)lan thu "<<i+1<<" :"<<exp(pred)-1<<endl;
        cout<<"thuc te:"<<fibo(i+4)<<endl;
    }
    
    //cout<<"predict(ko chuan hoa):"<<model.predictOutput_S(p_thu)*lon<<endl;
    return 0;
}